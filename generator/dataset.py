import os
import json
import logging
from copy import deepcopy

import torch
import torch.utils.data as data

import common.utils as utils
from common.vocab import Vocab
import common.Constants as Constants
from treeLSTM.treeNode import TreeNode
from common.QAConfig import QAConfig


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
info = logger.info


class CommonDataset(data.Dataset):
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sents = deepcopy(self.sentences[index])
        trees = deepcopy(self.trees[index])
        paths = self.data[index]['paths']
        targets = self.data[index]['targets']
        return sents, trees, paths, targets

    def __init__(self, path, vocab, device):
        super(CommonDataset, self).__init__()
        self.vocab = vocab
        self.device = device

        self.pov_data = self.read_data(os.path.join(path, 'pos.json'))
        self.neg_data = self.read_data(os.path.join(path, 'neg.json'))

        self.data = self.set_label()
        self.sentences = self.read_sentences()
        self.trees = self.read_trees()
        self.get_target()


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
        return data

    def get_target(self):
        for data_case in self.data:
            targets = []
            id_paths = []
            for path_case in data_case['paths']:
                target = []
                id_path = []
                for path in path_case:
                    path.insert(0, '<s>')
                    # path.append('</s>')
                    indices = self.vocab.convertToIdx(path, Constants.UNK_WORD)
                    # print('INDICES', indices)
                    id_path.append(indices)
                    end_id = indices[-1]
                    target.append(end_id)
                # one_hot_target = np.zeros([self.vocab.size(), len(path_case)])
                # one_hot_target[target, np.arange(len(path_case))] = 1
                id_paths.append(id_path)
                # targets.append(one_hot_target)
                targets.append(target)
            data_case['targets'] = targets
            data_case['paths'] = id_paths

    def build_vocab(self):
        utils.build_vocab([self.path], QAConfig.vocab)
        return Vocab(filename=QAConfig.vocab,
                     data=[
                         Constants.PAD_WORD, Constants.UNK_WORD,
                         Constants.BOS_WORD, Constants.EOS_WORD])

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
                    if i not in Nodes.keys():
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
                if root is None:
                    print(tri_case)
                tree_list.append(root)
            trees.append(tree_list)
        return trees