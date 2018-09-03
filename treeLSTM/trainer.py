from __future__ import division
from __future__ import print_function

import os
import sys
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
import torch.nn.functional as F

from treeLSTM.metrics import Metrics
from treeLSTM.dataset import IMDBdataset
from treeLSTM.model import TreeLSTM, ChildSumTreeLSTM

from common import utils
from common import Constants
from common.vocab import Vocab
from common.Global import Global
from common.argConfig import parse_args
from common.classificationConfig import classificationConfig


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        batch_loss = 0
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')
        for idx in tqdm(range(int(len(dataset))), desc='Training epoch ' + str(self.epoch + 1) + ''):
            sents, trees, label = dataset[indices[idx]]
            label = label.to(self.device)
            #print('LABEL', label.item())
            #print('SENTS:', sents)
            #print('TREES:', trees)
            #print('LABEL:', label)
            # target = utils.map_label_to_target(label, dataset.num_classes)
            # target = target.to(self.device)
            hiddens, output = self.model(sents, trees)
            # print('OUTPUT:', output)
            pred = F.softmax(output)
            pred = torch.max(pred, 0)
            # print('PRED:', pred)
            # print('root lngth:', len((hiddens)))
            # print('sent lngth:', len((sents)))
            # print('trees :', (trees))
            # print('hidden:', output.hidden)
            loss = self.criterion(output, label.unsqueeze(0))
            # print('LOSS:', loss)
            total_loss += loss.item()
            batch_loss += loss.data[0]
            # loss.backward()
            # print('BATCH LOSS:', batch_loss)
            if idx % 20 == 0 and idx > 0:
                print('BATCH LOSS', batch_loss.item())
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                batch_loss = 0
        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        right_num = 0
        with torch.no_grad():
            # total_loss = 0.0
            # predictions = torch.zeros(len(dataset), dtype=torch.float, device='cpu')
            indices = torch.arange(1, dataset.num_classes + 1, dtype=torch.float, device='cpu')
            for idx in tqdm(range(int(len(dataset))), desc='Testing epoch  ' + str(self.epoch) + ''):
                sents, trees, label = dataset[idx]
                label = label.to(self.device)
                #print('LABEL', label.item())
                # target = utils.map_label_to_target(label, dataset.num_classes)
                sents = [sent.to(self.device) for sent in sents]
                # target = target.to(self.device)
                hiddens, output = self.model(sents, trees)
                # print('ROOTs', hiddens)
                pred = F.softmax(output)
                pred = torch.max(pred, 1)[1]
                #print('LABEL', label)
                #print('PRED', pred)
                if pred.item() == label.item():
                    right_num = right_num + 1
                # loss = self.criterion(output, label.unsqueeze(0))
                # total_loss += loss.item()
                # output = output.squeeze().to('cpu')

                # predictions[idx] = torch.dot(indices, torch.exp(output))
        return right_num / len(dataset)# , predictions


if __name__ == '__main__':

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

    ## build vocab
    token_files = []
    for k in ['pos', 'neg']:
        token_files.extend( [os.path.join(token_file_label, k+".json") for token_file_label in classificationConfig.token_file_labels] )
    # imdb_vocab_file = os.path.join(args.data, 'imdb.vocab')
    print('token_files', token_files)
    imdb_vocab_file = classificationConfig.vocab
    utils.build_vocab(token_files, imdb_vocab_file)
    # get vocab object from vocab file previously written
    vocab = Vocab(filename=imdb_vocab_file,
                  data=[Constants.PAD_WORD, Constants.UNK_WORD,
                        Constants.BOS_WORD, Constants.EOS_WORD])
    logger.debug('==> imdb vocabulary size : %d ' % vocab.size())

    ## build embedding of vocab
    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    # emb_file = os.path.join(Global.external_tools, 'imdb_embed.pth')
    emb_file = classificationConfig.embed
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = utils.load_word_vectors(classificationConfig.glove)
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
    train_dir = classificationConfig.token_file_labels[0]
    train_file = os.path.join(Global.external_tools, 'imdb_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = IMDBdataset(train_dir, vocab, args.num_classes)
        torch.save(train_dataset, train_file)
        # train_dataset = torch.load(train_file)
    logger.debug('==> Size of train data   : %d ' % len(train_dataset))

    dev_dir = classificationConfig.token_file_labels[1]
    dev_file = os.path.join(Global.external_tools, 'imdb_dev.pth')
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = IMDBdataset(dev_dir, vocab, args.num_classes)
        torch.save(dev_dataset, dev_file)
    # dev_dataset = torch.load(dev_file)
    logger.debug('==> Size of dev data     : %d ' % len(dev_dataset))

    test_dir = classificationConfig.token_file_labels[2]
    test_file = os.path.join(Global.external_tools, 'imdb_test.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
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
        args.freeze_embed,
        device)
    criterion = nn.CrossEntropyLoss()

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
        # train_dataset = train_dataset[0:1]
        train_loss = trainer.train(train_dataset)
        train_accu = trainer.test(train_dataset)
        dev_accu = trainer.test(dev_dataset)
        test_accu = trainer.test(test_dataset)

        # train_pearson = metrics.pearson(train_pred, train_dataset.labels)
        # train_mse = metrics.mse(train_pred, train_dataset.labels)
        logger.info('==> Epoch {}, Train \tLoss: {}\tAccuracy: {}'.format(
            epoch, train_loss, train_accu))
        # dev_pearson = metrics.pearson(dev_pred, dev_dataset.labels)
        # dev_mse = metrics.mse(dev_pred, dev_dataset.labels)
        logger.info('==> Epoch {}, Dev \tAccuracy: {}'.format(
            epoch, dev_accu))
        # test_pearson = metrics.pearson(test_pred, test_dataset.labels)
        # test_mse = metrics.mse(test_pred, test_dataset.labels)
        logger.info('==> Epoch {}, Test \tAccuracy: {}'.format(
            epoch, test_accu))

        if best < dev_accu:
            best = dev_accu
            checkpoint = {
                'model': trainer.model.state_dict(),
                'optim': trainer.optimizer,
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

    g = Generator(QAConfig.tree_train)
    dataset = g.data_set
    print("start training")
    trainer = GeneratorTrainer(g, dataset, g.device)
    trainer.train()