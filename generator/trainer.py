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
from tqdm import tqdm

import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim

from common import utils
from common import Constants
from common.vocab import Vocab
from common.Global import Global
from common.argConfig import parse_args
from common.classificationConfig import classificationConfig

from generator.model import Generator
from generator.dataset import CommonDataset


class GeneratorTrainer:
    def __init__(self, model: Generator, dataset: CommonDataset, device):
        self.model = model
        self.dataset = dataset
        self.epoch = 10
        self.device = device
        self.criterion = self.model.seqback_criterion
        self.log_interval = 10
        self.optimizer = self.model.optimizer
        self.batch = 32

    # data refers to a sentence and corresponding properties
    def train(self):
        self.model.train()
        step = 0

        for i in range(self.epoch):
            for j in tqdm(range(len(self.dataset))):
                sentences, trees, paths, targets = self.dataset[j]
                for k in range(len(sentences)):
                    sentence = sentences[k]
                    print("sentence", self.dataset.vocab.convertToLabels(sentence, -100))
                    tree = trees[k]
                    target = targets[k]
                    target = np.array(target)
                    # path is torchized in model
                    path = paths[k]

                    target = torch.from_numpy(target).long().to(self.device)
                    sentence = [sentence.to(self.device)]
                    # tree is not need to be torchized
                    # tree = [tree.to(self.device)]
                    tree = [tree]
                    self.model.zero_grad()

                    output = self.model(sentence, tree, path)
                    loss = self.criterion(output, target)
                    loss.backward()
                    step += 1

                    softmax = nn.LogSoftmax(dim=1)
                    # print(output.shape)
                    # print(target.shape)
                    res = torch.argmax(softmax(output), 1)
                    # print("output", res)
                    # print("target", target)

                    print("output", self.dataset.vocab.convertToLabels(res, -100))
                    print("target", self.dataset.vocab.convertToLabels(target, -100))

                    if step % self.batch == 0:
                        print("LOSS", loss)
                        self.optimizer.step()
                        self.optimizer.zero_grad()


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

    # dev_dir = classificationConfig.token_file_labels[1]
    # dev_file = os.path.join(Global.external_tools, 'imdb_end2end_dev.pth')
    # if os.path.isfile(dev_file):
    #     dev_data = torch.load(dev_file)
    # else:
    #     dev_data = CommonDataset(dev_dir, vocab, device)
    #     torch.save(dev_data, dev_file)
    # logger.debug('==> Size of dev data     : %d ' % len(dev_data))

    train_dir = classificationConfig.token_file_labels[0]
    train_file = os.path.join(Global.external_tools, 'imdb_end2end_train.pth')
    if os.path.isfile(train_file):
        train_data = torch.load(train_file)
    else:
        train_data = CommonDataset(train_dir, vocab, device)
        torch.save(train_data, train_file)
    logger.debug('==> Size of train data   : %d ' % len(train_data))

    # test_dir = classificationConfig.token_file_labels[2]
    # test_file = os.path.join(Global.external_tools, 'imdb_end2end_test.pth')
    # if os.path.isfile(test_file):
    #     test_data = torch.load(test_file)
    # else:
    #     test_data = CommonDataset(test_dir, vocab, device)
    #     torch.save(test_data, test_file)
    # logger.debug('==> Size of test data    : %d ' % len(test_data))

    gen = Generator(vocab, emb, device)
    print("start training")
    trainer = GeneratorTrainer(gen, train_data, device)
    trainer.train()
