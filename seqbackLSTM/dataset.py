import os
import json
import codecs
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data

from treeLSTM.treeNode import TreeNode
import common.Constants as Constants


# Dataset class for IMDBdataset
class seqbackDataset(data.Dataset):
    def __init__(self, path, vocab, device, treeLSTM_model, bs=20):
        super(seqbackDataset, self).__init__()
        self.vocab = vocab
        self.device = device
        self.treelstm_model = treeLSTM_model
        self.batch_size = bs

        self.datafile_with_root = os.path.join(path, 'data.json')

        self.pov_data = self.read_data(os.path.join(path, 'pos.json'))
        self.neg_data = self.read_data(os.path.join(path, 'neg.json'))

        self.data = self.set_label()
        self.paths, self.targets = self.get_target()
        self.sentences = self.read_sentences()
        self.trees = self.read_trees()
        self.roots = self.get_root()

        self.save_data()

        self.size = len(self.roots)

        del self.data
        del self.sentences
        del self.trees

    def __getitem__(self, index):
        roots = deepcopy(self.roots[index])
        paths = deepcopy(self.paths[index])
        targets = deepcopy(self.targets[index])
        return roots, paths, targets

    def __len__(self):
        return self.size

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

    def read_sentences(self):
        sentences = []
        for data_case in self.data:
            sentence_list = []
            for sent_case in data_case['words']:
                indices = self.vocab.convertToIdx(sent_case, Constants.UNK_WORD)
                torch_indices = torch.tensor(indices, dtype=torch.long, device='cpu')
                sentence_list.append(torch_indices)
            sentences.append(sentence_list)
        print('READ SENTENCE DONE')
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
        print('READ TREE DONE')
        return trees

            # trees.append(tree_list)

    def get_target(self):
        id_paths = []
        id_targets = []
        d = 0
        for data_case in self.data:
            d = d+1
            print('PROCESS TARGET', d)
            targets = []
            paths = []
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
                paths.append(id_path)
                # targets.append(one_hot_target)
                targets.append(target)
            id_targets.append(targets)
            id_paths.append(paths)
        return id_paths, id_targets

    def get_root(self):
        if(len(self.sentences) != len(self.trees)):
            print('SENTENCES NOT MATCH TREES')
        roots = []
        d = 0
        self.treelstm_model.eval()
        with torch.no_grad():
            for (sentences, trees) in zip(self.sentences, self.trees):
                d = d+1
                print('GET ROOT FOR:', d)
                hiddens, _ = self.treelstm_model(sentences, trees)
                del _
                hiddens_to_save = [hidden.detach().cpu().numpy().tolist() for hidden in hiddens]
                del hiddens
                roots.append(hiddens_to_save)
                torch.cuda.empty_cache()
        return roots


    def save_data(self):
        seqback_data = {}
        seqback_data['targets'] = self.targets
        seqback_data['paths'] = self.paths
        seqback_data['roots'] = self.roots
        with codecs.open(self.datafile_with_root, "wb", 'utf-8') as f:
            json.dump(seqback_data, f, ensure_ascii=False, separators=(',', ':'))
