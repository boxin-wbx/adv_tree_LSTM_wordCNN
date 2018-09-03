from __future__ import division
from __future__ import print_function

import os
import time
import math
import json
import codecs
import random
import logging
import argparse
import numpy as np

import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim

from seqbackLSTM.dataset import seqbackDataset
from seqbackLSTM.model import SeqbackLSTM, chainLSTM
from treeLSTM.model import TreeLSTM, ChildSumTreeLSTM

from common import utils
from common import Constants
from common.vocab import Vocab
from common.Global import Global
from common.argConfig import parse_args
from common.classificationConfig import classificationConfig


class seqbackTrainer(object):
    def __init__(self, model, vocab, criterion, device, optimizer, bs=20, bptt=35, clip=0.25, log_interval=200):
        super(seqbackTrainer, self).__init__()
        self.model = model
        self.vocab = vocab
        self.criterion = criterion
        self.device = device
        self.optimizer = optimizer
        self.batch_size = bs
        self.bptt = bptt
        self.clip = clip
        self.log_interval = log_interval
        self.epoch = 0


    def repackage_hidden(self, original_root):
        root = np.array(original_root[0])
        root = torch.from_numpy(root)
        return root

    def get_data(self, train_data, i):
        sen_len = train_data[i]['length']
        paths = train_data[i]['paths']
        roots = train_data[i]['roots']
        targets = train_data[i]['targets']
        return sen_len, paths, roots, targets

    def evaluate(self, test_data):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        total_loss = 0.
        ntokens = self.vocab.size()
        with torch.no_grad():
            for batch, i in enumerate(range(0, len(test_data) - 1, self.bptt)):
                roots, paths, targets = test_data[i]
                for j in range(len(paths)):
                    root = self.repackage_hidden(roots[j])
                    data = paths[j]
                    target = targets[j]
                    target = np.array(target)
                    target = torch.from_numpy(target).long().to(self.device)

                    with codecs.open('seqback_datapip.json', "wb", 'utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))

                    self.model.zero_grad()
                    output = self.model(data, root)

                    softmax = nn.LogSoftmax(dim=1)
                    res = torch.argmax(softmax(output), 1)
                    print("output", self.vocab.convertToLabels(res, -100))
                    print("target", self.vocab.convertToLabels(target, -100))

                    loss = self.criterion(output, target)
                    total_loss = total_loss + loss

        return total_loss / len(test_data)


    def train(self, train_data, lr):
        # Turn on training mode which enables dropout.
        self.model.train()
        total_loss = 0.
        self.model.zero_grad()
        start_time = time.time()
        ntokens = self.vocab.size()
        for batch, i in enumerate(range(0, len(train_data) - 1, self.bptt)):
            # sen_len, paths, roots, targets = self.get_data(train_data, i)
            roots, paths, targets = train_data[i]
            for j in range(len(paths)):
                # print('PATHS:', paths)
                root = self.repackage_hidden(roots[j])
                target = targets[j]
                data = paths[j]
                # print('TARGET', len(target))
                target = np.array(target)
                target = torch.from_numpy(target).long().to(self.device)

                with codecs.open('seqback_datapip.json', "wb", 'utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, separators=(',', ':'))

                output = self.model(data, root)
                
                softmax = nn.LogSoftmax(dim=1)
                res = torch.argmax(softmax(output), 1)

                # print("output", self.vocab.convertToLabels(res, -100))
                # print("target", self.vocab.convertToLabels(target, -100))

                loss = self.criterion(output, target)
                print('LOSS', loss.data[0])
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        # clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
        #print('model parameters:', self.model.parameters())
        for p in self.model.parameters():
            if (p.requires_grad == True):
                p.data.add_(-lr, p.grad.data)

            total_loss += loss.item()

            if batch % self.log_interval == 0 and batch > 0:
                cur_loss = total_loss / self.log_interval
                elapsed = time.time() - start_time
                print('| {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    batch, len(train_data) // self.bptt, lr,
                    elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    def attack(self, data_source):
        # Turn on evaluation mode which disables dropout.
        sen_len, paths, roots, targets = self.get_data(data_source, 0)
        j = 0
        root = self.repackage_hidden(roots[j])
        print('FIRST ROOT', root)
        print('FIRST ROOT TYPE', root.size())
        root = torch.unsqueeze(root, 0)
        root = torch.unsqueeze(root, 0)
        print('FIRST ROOT', root)
        print('FIRST ROOT TYPE', root.size())

        data = paths[j]
        with codecs.open('seqback_datapip.json', "wb", 'utf-8') as f:
            json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
        # length = np.array([len(path) for path in data]) 
        target = np.zeros((1, 2))
        target[0, 0] = 1
        target[0, 1] = 0

        return root, target


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

    # get vocab object from vocab file previously written
    imdb_vocab_file = classificationConfig.vocab
    vocab = Vocab(filename=imdb_vocab_file,
                  data=[Constants.PAD_WORD, Constants.UNK_WORD,
                        Constants.BOS_WORD, Constants.EOS_WORD])
    logger.debug('==> imdb vocabulary size : %d ' % vocab.size())
    emb_file = classificationConfig.embed
    emb = torch.load(emb_file)

    ## built treeLSTM model
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
    tree_model.emb.weight.data.copy_(emb)
    with open('%s.pt' % os.path.join(args.save, args.expname), 'rb') as f:
        tree_model.load_state_dict(torch.load(f)['model'])
    tree_model.eval()

    # build dataset for seqbackLSTM
    # train_dir = classificationConfig.token_file_labels[0]
    # seqback_train_file = os.path.join(Global.external_tools, 'imdb_seqback_train.pth')
    # if os.path.isfile(seqback_train_file):
        # seqback_train_data = torch.load(seqback_train_file)
    # else:
        # seqback_train_data = seqbackDataset(train_dir, vocab, device, tree_model)
        # torch.save(seqback_train_data, seqback_train_file)
        # logger.debug('==> Size of train data   : %d ' % len(seqback_train_data))

    # test_dir = classificationConfig.token_file_labels[2]
    # seqback_test_file = os.path.join(Global.external_tools, 'imdb_seqback_test.pth')
    # if os.path.isfile(seqback_test_file):
        # seqback_test_data = torch.load(seqback_test_file)
    # else:
        # seqback_test_data = seqbackDataset(test_dir, vocab, device, tree_model)
        # torch.save(seqback_test_data, seqback_test_file)
    # logger.debug('==> Size of test data    : %d ' % len(seqback_test_data))

    dev_dir = classificationConfig.token_file_labels[1]
    seqback_dev_file = os.path.join(Global.external_tools, 'imdb_seqback_dev.pth')
    if os.path.isfile(seqback_dev_file):
        seqback_dev_data = torch.load(seqback_dev_file)
    else:
        seqback_dev_data = seqbackDataset(dev_dir, vocab, device, tree_model)
        torch.save(seqback_dev_data, seqback_dev_file)
    logger.debug('==> Size of dev data     : %d ' % len(seqback_dev_data))

    ## build seqbackLSTM model
    seqback_criterion = nn.CrossEntropyLoss()
    seqback_model = SeqbackLSTM(vocab, device)
    seqback_model.to(device), seqback_criterion.to(device)
    seqback_model.emb.weight.data.copy_(emb)

    # load the best saved seqback_model.
    with open(args.save_seqback, 'rb') as f:
        seqback_model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        seqback_model.chainLSTM.lstm.flatten_parameters()

    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      seqback_model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,
                                         seqback_model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                     seqback_model.parameters()), lr=args.lr, weight_decay=args.wd)

    ## train seqbackLSTM model
    seqback_trainer = seqbackTrainer(seqback_model, vocab, seqback_criterion, device, optimizer)
    # lr = 20
    # best_val_loss = None
    # for epoch in range(1, args.epochs+1):
        # epoch_start_time = time.time()
        # print('EPOCH:', epoch)
        # seqback_trainer.train(seqback_train_data, lr)
        # val_loss = seqback_trainer.evaluate(seqback_dev_data)
        # print('-' * 89)
        # print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                # 'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           # val_loss, math.exp(val_loss)))
        # print('-' * 89)
        # if not best_val_loss or val_loss < best_val_loss:
            # with open(args.save_seqback, 'wb') as f:
                # torch.save(seqback_model, f)
            # best_val_loss = val_loss
        # else:
            # lr /= 4.0




    ## SeqbackLSTM run on test data.
    test_loss = seqback_trainer.evaluate(seqback_dev_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)