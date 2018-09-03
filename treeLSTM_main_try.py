from __future__ import division
from __future__ import print_function

import os
import time
import math
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf

# CONFIG PARSER
from treeLSTM_config import parse_args

# UTILITY FUNCTIONS
from utils import utils
# IMPORT CONSTANTS
from utils import Constants
# DATA HANDLING CLASSES
from utils import Vocab

# NEURAL NETWORK MODULES/LAYERS
from treelstm import TreeLSTM
from treelstm import IMDBdataset
from treelstm import Metrics
from treelstm import Trainer

# IMPORT seqbackLSTM trainer, dataset, model
from seqbacklstm import seqbackTrainer
from seqbacklstm import seqbackDataset
from seqbacklstm import SeqbackLSTM

# IMPORT wordCNN trainer, evaluator dataset, model
from wordcnn import WordCNNTrainer
from wordcnn import WordCNNEvaluator
from wordcnn import WordCNNDataset
from wordcnn import WordCNNDataLoader
from wordcnn import WordCNN

# IMPORT CW related functions
from cleverhans.model import CallableModelWrapper
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf


# MAIN BLOCK
def main():

    ## setup experimental preparation
    global args
    args = parse_args()
    # built save folder
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    fh = logging.FileHandler(os.path.join(args.save, args.expname)+'.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # GPU select
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if args.sparse and args.wd != 0:
        logger.error('Sparsity and weight decay are incompatible, pick one!')
        exit()
    # debugging args
    logger.debug(args)
    # set seed for 
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    # datadet file
    train_dir = os.path.join(args.data, 'aclImdb/train/')
    dev_dir = os.path.join(args.data, 'aclImdb/dev/')
    test_dir = os.path.join(args.data, 'aclImdb/test/')
    token_file_labels = [dev_dir, train_dir, test_dir]

    ## processe_raw_data
    # deal with IMDB dataset: sentence to tree
    # for token_file_label in token_file_labels:
    #     utils.processe_raw_data(token_file_label)

    ## build vocab
    token_files = []
    for k in ['pos', 'neg']:
        token_files.extend( [os.path.join(token_file_label, k+".json") for token_file_label in token_file_labels] )
    imdb_vocab_file = os.path.join(args.data, 'imdb.vocab')
    utils.build_vocab(token_files, imdb_vocab_file)
    # get vocab object from vocab file previously written
    vocab = Vocab(filename=imdb_vocab_file,
                  data=[Constants.PAD_WORD, Constants.UNK_WORD,
                        Constants.BOS_WORD, Constants.EOS_WORD])
    logger.debug('==> imdb vocabulary size : %d ' % vocab.size())

    ## build embedding of vocab
    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'imdb_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = utils.load_word_vectors(os.path.join(args.glove, 'glove.840B.300d'))
        logger.debug('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
        emb = torch.zeros(vocab.size(), glove_emb.size(1), dtype=torch.float, device=device)
        emb.normal_(0, 0.05)
        # zero out the embeddings for padding and other special words if they are absent in vocab
        for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD,
                                    Constants.BOS_WORD, Constants.EOS_WORD]):
            if idx == 0:
                emb[idx].fill_(10e-3)
            if idx == 1:
                emb[idx].fill_(10e-1)
            if idx == 2:
                emb[idx].fill_(1)
            if idx == 3:
                emb[idx].fill_(2)
        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
        torch.save(emb, emb_file)

    ## build dataset for treelstm
    # load imdb dataset splits
    train_file = os.path.join(args.data, 'imdb_train.pth')
    train_dataset = IMDBdataset(train_dir, vocab, args.num_classes)
    torch.save(train_dataset, train_file)
    # train_dataset = torch.load(train_file)
    logger.debug('==> Size of train data   : %d ' % len(train_dataset))

    dev_file = os.path.join(args.data, 'imdb_dev.pth')
    dev_dataset = IMDBdataset(dev_dir, vocab, args.num_classes)
    torch.save(dev_dataset, dev_file)
    # dev_dataset = torch.load(dev_file)
    logger.debug('==> Size of dev data     : %d ' % len(dev_dataset))

    test_file = os.path.join(args.data, 'imdb_test.pth')
    test_dataset = IMDBdataset(test_dir, vocab, args.num_classes)
    torch.save(test_dataset, test_file)
    # test_dataset = torch.load(test_file)
    logger.debug('==> Size of test data    : %d ' % len(test_dataset))

    ## built treeLSTM model
    # initialize tree_model, criterion/loss_function, optimizer
    tree_model = TreeLSTM(
        vocab.size(),
        args.input_dim,
        args.mem_dim,
        args.hidden_dim,
        args.num_classes,
        args.sparse,
        args.freeze_embed)
    criterion = nn.KLDivLoss()

    tree_model.to(device), criterion.to(device)
    # plug these into embedding matrix inside tree_model
    tree_model.emb.weight.data.copy_(emb)

    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      tree_model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,
                                         tree_model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                     tree_model.parameters()), lr=args.lr, weight_decay=args.wd)
    metrics = Metrics(args.num_classes)

    ## train treeLSTM model
    # create trainer object for training and testing
    trainer = Trainer(args, tree_model, criterion, optimizer, device)
    best = -float('inf')
    for epoch in range(args.epochs):
        train_loss = trainer.train(train_dataset)
        train_loss, train_pred = trainer.test(train_dataset)
        dev_loss, dev_pred = trainer.test(dev_dataset)
        test_loss, test_pred = trainer.test(test_dataset)

        train_pearson = metrics.pearson(train_pred, train_dataset.labels)
        train_mse = metrics.mse(train_pred, train_dataset.labels)
        logger.info('==> Epoch {}, Train \tLoss: {}\tPearson: {}\tMSE: {}'.format(
            epoch, train_loss, train_pearson, train_mse))
        dev_pearson = metrics.pearson(dev_pred, dev_dataset.labels)
        dev_mse = metrics.mse(dev_pred, dev_dataset.labels)
        logger.info('==> Epoch {}, Dev \tLoss: {}\tPearson: {}\tMSE: {}'.format(
            epoch, dev_loss, dev_pearson, dev_mse))
        test_pearson = metrics.pearson(test_pred, test_dataset.labels)
        test_mse = metrics.mse(test_pred, test_dataset.labels)
        logger.info('==> Epoch {}, Test \tLoss: {}\tPearson: {}\tMSE: {}'.format(
            epoch, test_loss, test_pearson, test_mse))

        if best < test_pearson:
            best = test_pearson
            checkpoint = {
                'model': trainer.model.state_dict(),
                'optim': trainer.optimizer,
                'pearson': test_pearson, 'mse': test_mse,
                'args': args, 'epoch': epoch
            }
            logger.debug('==> New optimum found, checkpointing everything now...')
            torch.save(checkpoint, '%s.pt' % os.path.join(args.save, args.expname))


    ## get the tree root note position of every sentence
    with open('%s.pt' % os.path.join(args.save, args.expname), 'rb') as f:
        tree_model.load_state_dict(torch.load(f)['model'])
    datasets= [train_dataset, test_dataset, dev_dataset]
    for dataset in datasets:
        dataset.get_root(tree_model, device)
    # for dataset in datasets:
    #     indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')
    #     for idx in tqdm(range(len(dataset)), desc='Building root representation...'):
    #         sents, trees, _ = dataset[indices[idx]]
    #         # print('SENTS:', sents)
    #         # print('TREES:', trees)
    #         sents = [sent.to(device) for sent in sents]
    #         hiddens, _ = tree_model(sents, trees)
    #         print('ROOTS:', hiddens)
    #         # print('TO ADD:', dataset[indices[idx]])
    #         dataset[indices[idx]][0].append(hiddens)
    #         # print('TO ADD:', dataset[indices[idx]])

    ## build dataset for seqbackLSTM
    seqback_train_file = os.path.join(args.data, 'imdb_seqback_train.pth')
    # seqback_train_dataset = seqbackDataset(train_dir, vocab, device).sequenses
    seqback_train_data = seqbackDataset(train_dir, vocab, device)
    torch.save(seqback_train_data, seqback_train_file)
    # seqback_train_dataset = torch.load(seqback_train_file)
    logger.debug('==> Size of train data   : %d ' % len(seqback_train_data))

    seqback_val_file = os.path.join(args.data, 'imdb_seqback_dev.pth')
    # seqback_val_dataset = seqbackDataset(dev_dir, vocab, device).sequenses
    seqback_val_data = seqbackDataset(dev_dir, vocab, device)
    torch.save(seqback_val_data, seqback_val_file)
    # seqback_dev_dataset = torch.load(seqback_dev_file)
    logger.debug('==> Size of dev data     : %d ' % len(seqback_val_data))

    seqback_test_file = os.path.join(args.data, 'imdb_seqback_test.pth')
    # seqback_test_dataset = seqbackDataset(test_dir, vocab, device).sequenses
    seqback_test_data = seqbackDataset(test_dir, vocab, device)
    torch.save(seqback_test_data, seqback_test_file)
    # seqback_test_dataset = torch.load(seqback_test_file)
    logger.debug('==> Size of test data    : %d ' % len(seqback_test_data))

    ## build seqbackLSTM model
    seqback_criterion = nn.CrossEntropyLoss()
    seqback_model = SeqbackLSTM(vocab, device)
    seqback_model.to(device), seqback_criterion.to(device)
    seqback_model.emb.weight.data.copy_(emb)

    ## train seqbackLSTM model
    seqback_trainer = seqbackTrainer(seqback_model, vocab, seqback_criterion, device, optimizer)
    lr = 20
    best_val_loss = None
    # At any point you can hit Ctrl + C to break out of training early.
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        print('EPOCH:', epoch)
        seqback_trainer.train(seqback_train_data, lr)
        val_loss = seqback_trainer.evaluate(seqback_val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save_seqback, 'wb') as f:
                torch.save(seqback_model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
    # load the best saved seqback_model.
    with open(args.save_seqback, 'rb') as f:
        seqback_model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        seqback_model.chainLSTM.lstm.flatten_parameters()

    ## SeqbackLSTM run on test data.
    test_loss = seqback_trainer.evaluate(seqback_test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    ## build dataset of wordCNN
    wordcnn_train_file = os.path.join(args.data, 'imdb_wordcnn_train.pth')
    wordcnn_train_data = WordCNNDataset(train_dir, vocab, device).Preprocessor
    wordcnn_train_dataloader = WordCNNDataLoader(dataset=wordcnn_train_data, batch_size=64)
    torch.save(wordcnn_train_dataloader, wordcnn_train_file)
    logger.debug('==> Size of train data   : %d ' % len(wordcnn_train_data))

    wordcnn_val_file = os.path.join(args.data, 'imdb_wordcnn_dev.pth')
    wordcnn_val_data = WordCNNDataset(dev_dir, vocab, device).Preprocessor
    wordcnn_val_dataloader = WordCNNDataLoader(dataset=wordcnn_val_data, batch_size=64)
    torch.save(wordcnn_val_dataloader, wordcnn_val_file)
    logger.debug('==> Size of dev data     : %d ' % len(wordcnn_val_data))

    wordcnn_test_file = os.path.join(args.data, 'imdb_wordcnn_test.pth')
    wordcnn_test_data = WordCNNDataset(test_dir, vocab, device).Preprocessor
    wordcnn_test_dataloader = WordCNNDataLoader(dataset=wordcnn_test_data, batch_size=64)
    torch.save(wordcnn_test_dataloader, wordcnn_test_file)
    logger.debug('==> Size of test data    : %d ' % len(wordcnn_test_data))

    wordcnn_model = WordCNN(2, vocab, emb)
    wordcnn_model.to(device)

    trainable_params = [p for p in wordcnn_model.parameters() if p.requires_grad]
    wordcnn_optimizer = optim.Adam(params=trainable_params, lr=0.01)
    # wordcnn_optimizer = Adadelta(params=trainable_params, lr=0.01, weight_decay=0.95)
    lr_plateau = optim.lr_scheduler.ReduceLROnPlateau(wordcnn_optimizer, factor=0.7, patience=5, min_lr=0.0001)
    wordcnn_criterion = nn.CrossEntropyLoss
    wordcnn_trainer = WordCNNTrainer(wordcnn_model, wordcnn_train_dataloader, wordcnn_val_dataloader, 
                      criterion=wordcnn_criterion, optimizer=wordcnn_optimizer, 
                      lr_schedule='store_true', lr_scheduler=lr_plateau, use_gpu=torch.cuda.is_available(), logger=logger)
    wordcnn_trainer.run(epochs=10)

    logger.info("Evaluating...")
    logger.info('Best Model: {}'.format(wordcnn_trainer.best_checkpoint_filepath))
    wordcnn_model.load_state_dict(torch.load(wordcnn_trainer.best_checkpoint_filepath))
    wordcnn_evaluator = WordCNNEvaluator(wordcnn_model, wordcnn_test_dataloader, use_gpu=torch.cuda.is_available(), logger=logger)
    wordcnn_evaluator.evaluate()


    ## Craft adversarial examples using Carlini and Wagner's approach

    # nn.sequential to Merge seqbackLSTM and wordCNN
    


    nb_classes = 2
    source_samples=10
    sess = tf.Session()
    x_op = tf.placeholder(tf.float32, shape=(None, 1, 28, 28,))

    # Convert pytorch model to a tf_model and wrap it in cleverhans
    seqback_model.train()
    wordcnn_model.train()
    seqbacklstm_and_wordcnn_model = torch.nn.Sequential(seqback_model, wordcnn_model)
    tf_seqbacklstm_and_wordcnn_model = convert_pytorch_model_to_tf(seqbacklstm_and_wordcnn_model)
    chans_tf_seqbacklstm_and_wordcnn_model = CallableModelWrapper(tf_seqbacklstm_and_wordcnn_model, output_layer='logits')
    # tf_seqback_model = convert_pytorch_model_to_tf(seqback_model)
    #cleverhans_model1 = CallableModelWrapper(tf_model1, output_layer='logits')
    # tf_wordcnn_model = convert_pytorch_model_to_tf(wordcnn_model)
    #cleverhans_model2 = CallableModelWrapper(tf_model2, output_layer='logits')
    # cleverhans_model = torch.nn.Sequential(tf_model1, tf_model2)
    # cleverhans_model = CallableModelWrapper(cleverhans_model, output_layer='logits')

    # CW model
    cw = CarliniWagnerL2(chans_tf_seqbacklstm_and_wordcnn_model, back='tf', sess=sess)

    # build adv_inputs
    #adv_inputs = np.array([[instance] * nb_classes for instance in x_test[:source_samples]], dtype=np.float32)
    #adv_inputs = adv_inputs.reshape((source_samples * nb_classes, img_rows, img_cols, nchannels))

    #one_hot = np.zeros((nb_classes, nb_classes))
    #one_hot[np.arange(nb_classes), np.arange(nb_classes)] = 1
    #adv_ys = np.array([one_hot] * source_samples, dtype=np.float32).reshape((source_samples * nb_classes, nb_classes))
    yname = "y_target"
    adv_inputs, adv_ys = seqback_trainer.attack(seqback_test_data)

    cw_params = {'binary_search_steps': 1, yname: adv_ys,
                 'max_iterations': 100,
                 'learning_rate': 0.1,
                 'batch_size': 1,
                 'initial_const': 10}

    adv = cw.generate_np(adv_inputs, **cw_params)

    print('ROOT ADV', adv)

    eval_params = {'batch_size': np.minimum(nb_classes, source_samples)}

    adv_accuracy = model_eval(sess, x, y, preds, adv, adv_ys, args=eval_params)

    # Compute the number of adversarial examples that were successfully found
    print('Avg. rate of successful adv. examples {0:.4f}'.format(adv_accuracy))

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(np.sum((adv - adv_inputs)**2, axis=(1, 2, 3))**.5)
    print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))

    # Close TF session
    sess.close()



if __name__ == "__main__":
    main()
