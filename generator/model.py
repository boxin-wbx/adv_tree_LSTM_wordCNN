import os
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from common.argConfig import parse_args

from treeLSTM.model import TreeLSTM
from seqbackLSTM.model import SeqbackLSTM

from generator.dataset import CommonDataset

args = parse_args()

class Generator(nn.Module):
    # sents/trees/paths are specific to one sentence
    def forward(self, sents, trees, paths):
        hiddens, prediction = self.tree_model(sents, trees)
        return self.seqback_model(paths, hiddens[0])

    def __init__(self, vocab, embed, device):
        super().__init__()

        self.vocab = vocab
        self.device = device
        self.embed = embed

        # set seed for embedding metrics
        torch.manual_seed(args.seed)
        random.seed(args.seed)

        # initialize tree_model, criterion/loss_function, optimizer
        self.tree_model = TreeLSTM(
            self.vocab.size(),
            args.input_dim,
            args.mem_dim,
            args.hidden_dim,
            args.num_classes,
            args.sparse,
            args.freeze_embed,
            device=self.device)

        self.tree_criterion = nn.KLDivLoss()
        # todo: tree criterion might be useless
        self.tree_model.to(self.device), self.tree_criterion.to(self.device)
        # plug these into embedding matrix inside tree_model
        self.tree_model.emb.weight.data.copy_(self.embed)

        if args.optim == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                               self.tree_model.parameters()), lr=args.lr, weight_decay=args.wd)
        elif args.optim == 'adagrad':
            self.optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,
                                                  self.tree_model.parameters()), lr=args.lr, weight_decay=args.wd)
        elif args.optim == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                              self.tree_model.parameters()), lr=args.lr, weight_decay=args.wd)

        self.seqback_model = SeqbackLSTM(self.vocab, self.device)
        self.seqback_criterion = nn.CrossEntropyLoss()
        self.seqback_model.to(self.device), self.seqback_criterion.to(self.device)
        self.seqback_model.emb.weight.data.copy_(self.embed)

        # logger.debug('==> Size of train data   : %d ' % len(self.data_set))
