import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import common.Constants as Constants


# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state


# putting the whole model together
class TreeLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity, freeze, device="cpu"):
        super(TreeLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        self.device = device
        if freeze:
            self.emb.weight.requires_grad = False
        self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim)
        # self.wh = nn.Linear(mem_dim, hidden_dim)
        self.wp = nn.Linear(hidden_dim, num_classes)

    def forward(self, sents, trees):
        length = len(list(sents))
        # print(length)
        sents = [sent.to(self.device) for sent in sents]
        prediction = torch.zeros(1, 2).to(self.device)

        # print(prediction)
        # print(type(prediction))
        # print(prediction.type())
        hiddens = []
        # print(len(sents))
        # print(sents)
        # print(len(trees))
        # print(trees)
        for (sent, tree) in zip(sents, trees):
            # print('SENT:', sent)
            # print('TREE:', tree)
            if tree == None:
                continue
            sent = self.emb(sent)
            state, hidden = self.childsumtreelstm(tree, sent)
            # print('STATE', state)
            # print('HIDDEN', hidden)
            hiddens.append(hidden)
            # pred = F.sigmoid(self.wh(hidden))
            # print('PRED', pred)
            pred = self.wp(hidden)
            # pred = F.softmax(self.wp(hidden), dim=1)
            # print('PRED', pred)
            # print(pred)
            # print(type(pred))
            # print(pred.type())
            # todo: change to cuda version
            # prediction = torch.add(prediction, pred).cuda()
            prediction = torch.add(prediction, pred.detach())
        prediction = torch.div(prediction, length)
        # print('Prediction', prediction)
        return hiddens, prediction
