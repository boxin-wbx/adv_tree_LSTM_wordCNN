from __future__ import division
from __future__ import print_function

import os
import re
import html
import nltk
import math
import json
import torch
import string
import codecs
import bleach
import logging

from tqdm import tqdm

from .vocab import Vocab
# from common.vocab import Vocab
from nltk.parse.corenlp import CoreNLPDependencyParser
# from common.QAConfig import QAConfig

logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
info = logger.info

punctuations = list(string.punctuation)
punctuations.append('...')


# loading GLOVE word vectors
# if .pth file is found, will load that
# else will load from .txt file & save
def load_word_vectors(path):
    if os.path.isfile(path + '.pth') and os.path.isfile(path + '.vocab'):
        print('==> File found, loading to memory')
        vectors = torch.load(path + '.pth')
        vocab = Vocab(filename=path + '.vocab')
        return vocab, vectors
    # saved file not found, read from txt file
    # and create tensors for word vectors
    print('==> File not found, preparing, be patient')
    count = sum(1 for line in open(path + '.txt', 'r', encoding='utf8', errors='ignore'))
    with open(path + '.txt', 'r') as f:
        contents = f.readline().rstrip('\n').split(' ')
        dim = len(contents[1:])
    words = [None] * (count)
    vectors = torch.zeros(count, dim, dtype=torch.float, device='cpu')
    with open(path + '.txt', 'r', encoding='utf8', errors='ignore') as f:
        idx = 0
        for line in f:
            contents = line.rstrip('\n').split(' ')
            words[idx] = contents[0]
            values = list(map(float, contents[1:]))
            vectors[idx] = torch.tensor(values, dtype=torch.float, device='cpu')
            idx += 1
    with open(path + '.vocab', 'w', encoding='utf8', errors='ignore') as f:
        for word in words:
            f.write(word + '\n')
    vocab = Vocab(filename=path + '.vocab')
    torch.save(vectors, path + '.pth')
    return vocab, vectors


# mapping from scalar to vector
def map_label_to_target(label, num_classes):
    target = torch.zeros(1, num_classes, dtype=torch.float, device='cpu')
    ceil = int(math.ceil(label))
    floor = int(math.floor(label))
    if ceil == floor:
        target[0, floor - 1] = 1
    else:
        target[0, floor - 1] = ceil - label
        target[0, ceil - 1] = label - floor
    return target


def parse(sentence):
    parser = CoreNLPDependencyParser(url='http://localhost:9000')
    parse = parser.raw_parse(sentence)
    parse_tree = list(parse)[0]

    triple = []
    parse_values = []
    # fuck = False
    # fuck_checker = 0
    for k in parse_tree.nodes.values():
        if k is not None:
            parse_values.append(k)
        else:
            print("NONE happened", sentence)
        # if fuck_checker != k["address"]:
        #     fuck = True
        # fuck_checker += 1
    parse_values.sort(key=lambda x: x["address"])
    parse_values = parse_values[1:]
    words = [x["word"] for x in parse_values]
    # if fuck:
    #     print("sentence", sentence)
    print("word", words)

    for k in parse_tree.nodes.values():
        try:
            if k["address"] == 0:
                continue
            elif k["head"] == 0:
                triple.append((("ROOT", k["head"] - 1), (words[k["address"] - 1], k["address"] - 1), k["rel"]))
            else:
                triple.append(
                    ((words[k["head"] - 1], k["head"] - 1), (words[k["address"] - 1], k["address"] - 1), k["rel"]))
        except IndexError:
            print(words)
    return triple, words


def build_vocab(filepaths, dst_path):
    vocab = set()
    for filepath in filepaths:
        with codecs.open(filepath, "r", 'utf-8') as f:
            sen_tris = json.load(f, encoding='utf-8')
            for sen_tri in sen_tris:
                for word in sen_tri['words']:
                    vocab |= set(word)
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')


def build_path(triple, words):
    triple_dict = {}
    parents = []
    for tri in triple:
        triple_dict[tri[1][1]] = tri[0][1]
        parents.append(tri[0][1])
    parents = set(parents)
    paths = []
    for i, word in enumerate(words):
        if word not in [',', '.']:
            path = []
            now_node = i
            while now_node != -1:
                path.insert(0, words[now_node])
                # path.append(words[now_node])
                if now_node in triple_dict.keys():
                    now_node = triple_dict[now_node]
                else:
                    print('now_node', now_node)
                    print('Wrong triple_dict', triple_dict)
                    break;
            print('PATH', path)
            paths.append(path)
    return paths


def tokenize(paragraph):
    # cleaning HTMLs
    print('PARAGRAPH:', paragraph)
    paragraph = html.unescape(paragraph)
    paragraph = bleach.clean(paragraph, tags=[], strip=True).strip()
    paragraph = html.unescape(paragraph)
    # cutting sentences
    sentences = nltk.sent_tokenize(paragraph)
    # print('Sentence Length', len(sentences))

    sentence_words = []
    sentence_triples = []
    sentence_paths = []

    for sentence in sentences:
        if len(sentence) == 1:
            continue
        # print('Original Sentences:', sentence)
        # sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", sentence)
        sentence = re.sub('[^.,a-zA-Z0-9 \n\.]', '', sentence)
        sentence = sentence.lower()
        # print('Cleaned Sentence:', sentence)
        # words = nltk.word_tokenize(sentence)
        # print('Original words:', words)
        # punctuations = list(string.punctuation)
        # punctuations.append('...')
        # words = [i for i in words if i not in punctuations]
        # print('De-punctuation words:', words)
        # if words == []:
        #     continue
        # Build Dependency Tree
        try:
            tri, words = parse(sentence)
            if len(words) == 1:
                continue
            # print('Tri:', tri)
            # Build Tree Path
            path = build_path(tri, words)
            # print('Path', path)

            sentence_words.append(words)
            sentence_triples.append(tri)
            sentence_paths.append(path)
        except:
            print('DENPENCY TREE SENTE', sentence)
        # Tri-tree structure Word2ID：son->parent
        # tri_id = tri2id(tri)
        # print('Tri_id:', tri_id)
        # tree_one = read_tree(tri)
        # print('Tree', tree_one)
    return sentence_words, sentence_triples, sentence_paths


def generate_qa_input():
    # open articles
    with open(QAConfig.train_share, "r") as train_file:
        data = train_file.readline()
        train_set = json.loads(data)

    i = 0
    sen_tris = []
    for article in tqdm(train_set['p']):
        for paragraph in article:
            (sentence_words, sentence_triples, sentence_paths) = tokenize(paragraph)
            sen_tri = {'id': i}
            i += 1
            sen_tri['words'] = sentence_words
            sen_tri['triples'] = sentence_triples
            sen_tri['paths'] = sentence_paths
            sen_tris.append(sen_tri)
    with open(QAConfig.tree_train, "w") as f:
        json.dump(sen_tris, f)


def processe_raw_data(data_file):
    for j, label in enumerate(['pos', 'neg']):
        curdir = os.path.join(data_file, label)
        # outfile = os.path.join(datadir, '{0}-{1}.json'.format(i, j))

        info('reading {}'.format(curdir))
        sen_tris = []
        for k, elm in enumerate(os.listdir(curdir)):
            print('K and ELM:', k, elm)
            # if k > 0:
            #     break
            with open(os.path.join(curdir, elm), 'r') as r:
                sentence = re.sub('[ \t\n]+', ' ', r.read().strip())
                sentence = sentence.replace(".", ". ")
                sentence_words, sentence_triples, sentence_paths = tokenize(sentence)
                sen_tri = {}
                sen_tri['id'] = k
                sen_tri['words'] = sentence_words
                sen_tri['triples'] = sentence_triples
                sen_tri['paths'] = sentence_paths
                if(len(sentence_words) == len(sentence_triples) and len(sentence_triples) == len(sentence_paths)):
                    sen_tri['length'] = len(sentence_words)
                else:
                    sen_tri['length'] = None
                    print('WRONG SENTENCE', sentence)
                sen_tris.append(sen_tri)
        with codecs.open(os.path.join(data_file, label + ".json"), "wb", 'utf-8') as f:
            json.dump(sen_tris, f, ensure_ascii=False, separators=(',', ':'))


if __name__ == '__main__':
    generate_qa_input()