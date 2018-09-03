import os
import sys
import json
import codecs
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data

sys.path.append("..")
from common import Constants
from .treeNode import TreeNode


# Dataset class for IMDBdataset
class IMDBdataset(data.Dataset):
    def __init__(self, path, vocab, num_classes):
        super(IMDBdataset, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes

        self.pov_data = self.read_data(os.path.join(path, 'pos.json'))
        self.neg_data = self.read_data(os.path.join(path, 'neg.json'))
        self.datafile_with_root = os.path.join(path, 'data.json')
        # print('POV DATA:', self.pov_data)
        # print('NEG DATA:', self.neg_data)

        self.data = self.set_label()
        #print('ALL DATA:', self.data)

        self.sentences = self.read_sentences()
        self.trees = self.read_trees()
        self.labels = self.read_labels()

        self.size = self.labels.size(0)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        sent = deepcopy(self.sentences[index])
        tree = deepcopy(self.trees[index])
        label = deepcopy(self.labels[index])
        return sent, tree, label

    def read_data(self, filename):
        with open(filename,'r') as load_f:
            data = json.load(load_f)
        return data

    def set_label(self):
        data = []
        for data_one in self.pov_data:
            data_one['label'] = 0
            data.append(data_one)
        for data_one in self.neg_data:
            data_one['label'] = 1
            data.append(data_one)
        # print('ALL DATA:', data)
        return data

    def read_sentences(self):
        sentences = []
        for data_case in self.data:
            sentence_list = []
            for sent_case in data_case['words']:
                indices = self.vocab.convertToIdx(sent_case, Constants.UNK_WORD)
                torch_indices = torch.tensor(indices, dtype=torch.long, device='cpu')
                sentence_list.append(torch_indices)
            sentences.append(sentence_list)
        # with open(filename, 'r') as f:
        #     sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_trees(self):
        trees = []
        for data_case in self.data:
            tree_list = []
            for tri_case in data_case['triples']:
                tri_case.sort(key=lambda x: x[1][1])
                # print('Tri sorted:', tri_case)
                Nodes = dict()
                root = None
                for i in range(len(tri_case)):
                    # if i not in Nodes.keys() and tri_case[i][0][1] != -1:
                    if i not in Nodes.keys() and tri_case[i][0][1] != -1:
                        idx = i
                        prev = None
                        while True:
                            tree = TreeNode()
                            Nodes[idx] = tree
                            tree.idx = idx
                            if prev is not None:
                                tree.add_child(prev)
                            parent = tri_case[idx][0][1]
                            if parent in Nodes.keys():
                                Nodes[parent].add_child(tree)
                                break
                            elif parent == -1:
                                root = tree
                                break
                            else:
                                prev = tree
                                idx = parent
                tree_list.append(root)
            trees.append(tree_list)
        return trees


    def read_labels(self):
        labels = []
        for data_case in self.data:
            label = data_case['label']
            labels.append(label)
        # labels = list(map(lambda x: float(x), f.readlines()))
        labels = torch.tensor(labels, dtype=torch.long, device='cpu')
        return labels

    def get_root(self, model, device):
        for data_case in self.data:
            tree_list = []
            for tri_case in data_case['triples']:
                tri_case.sort(key=lambda x: x[1][1])
                # print('Tri sorted:', tri_case)
                Nodes = dict()
                root = None
                for i in range(len(tri_case)):
                    # if i not in Nodes.keys() and tri_case[i][0][1] != -1:
                    if i not in Nodes.keys() and tri_case[i][0][1] != -1:
                        idx = i
                        prev = None
                        while True:
                            tree = TreeNode()
                            Nodes[idx] = tree
                            tree.idx = idx
                            if prev is not None:
                                tree.add_child(prev)
                            parent = tri_case[idx][0][1]
                            if parent in Nodes.keys():
                                Nodes[parent].add_child(tree)
                                break
                            elif parent == -1:
                                root = tree
                                break
                            else:
                                prev = tree
                                idx = parent
                tree_list.append(root)
            sentence_list = []
            for sent_case in data_case['words']:
                indices = self.vocab.convertToIdx(sent_case, Constants.UNK_WORD)
                torch_indices = torch.tensor(indices, dtype=torch.long, device='cpu')
                sentence_list.append(torch_indices)
            sentence_list = [sent.to(device) for sent in sentence_list]
            hiddens, _ = model(sentence_list, tree_list)
            # print('Hidden', hiddens)
            data_case['roots'] = [hidden.detach().cpu().numpy().tolist() for hidden in hiddens]
        with codecs.open(self.datafile_with_root, "wb", 'utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, separators=(',', ':'))
