import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

from . import Constants


def pad_text(text, pad, min_length=None, max_length=None):
    length = len(text)
    if min_length is not None and length < min_length:
        return text + [pad]*(min_length - length)
    if max_length is not None and length > max_length:
        return text[:max_length]
    return text

# Dataset class for IMDBdataset
class WordCNNDataset(Dataset):
    def __init__(self, path, vocab, device, bs=64, min_length=5, max_length=300):
        super(WordCNNDataset, self).__init__()
        self.vocab = vocab
        self.device = device
        self.batch_size = bs
        self.min_length = min_length
        self.max_length = max_length

        self.PAD_IDX = Constants.PAD
        print('PAD_IDX:', self.PAD_IDX)

        self.pov_data = self.read_data(os.path.join(path, 'pos.json'))
        self.neg_data = self.read_data(os.path.join(path, 'neg.json'))
        # print('POV DATA:', self.pov_data)
        # print('NEG DATA:', self.neg_data)

        self.data = self.set_label()
        # print('ALL DATA:', self.data)

        self.Preprocessor = self.preprocessor()
        print('Preprocessor DATA[0]:', self.Preprocessor[0])
        print('Preprocessor DATA[0] type:', type(self.Preprocessor[0]))
        print('Preprocessor DATA type:', type(self.Preprocessor))

        self.size = len(self.Preprocessor)
        # self.sentences = self.read_sentences()
        # self.trees = self.read_trees()
        # self.labels = self.read_labels()

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.Preprocessor[index]


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

    def preprocessor(self):
        text_label_tuples = []
        for data_case in self.data:
            text = []
            for word_case in data_case['words']:
                indices = self.vocab.convertToIdx(word_case, Constants.UNK_WORD)
                text.extend(indices)
            padedtext = pad_text(text, self.PAD_IDX, self.min_length, self.max_length)
            label = data_case['label']
            text_label_tuples.append((padedtext, label))
        return text_label_tuples

class WordCNNDataLoader(DataLoader):
    
    def __init__(self, *args, **kwargs):
        super(WordCNNDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.PAD_IDX = Constants.PAD

    def __getitem__(self, index):
        texts_tensor, labels_tensor = self.collate_fn
        return texts_tensor[index], labels_tensor[index]

    def _collate_fn(self, batch):
        text_lengths = [len(text) for text, label in batch]
        
        longest_length = max(text_lengths)

        texts_padded = [pad_text(text, pad=self.PAD_IDX, min_length=longest_length) for text, label in batch]
        labels = [label for text, label in batch]
        
        texts_tensor, labels_tensor = torch.LongTensor(texts_padded), torch.LongTensor(labels)
        return texts_tensor, labels_tensor
