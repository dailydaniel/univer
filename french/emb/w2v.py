#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import argparse

from functools import reduce

import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F
import gensim

from scipy.spatial import distance
import gzip
import re
from collections import Counter, OrderedDict
from tqdm import tqdm
import pickle
import time

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default='opinrankdatasetwithjudgments.tar.gz', type=str)
    return parser

def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x

def read_input(input_file):
    """This method reads the input file which is in gzip format"""

    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            yield gensim.utils.simple_preprocess(line)

def ha(source):
    pattern = r'[ \'\(\)\[\]]'
    return '{}'.format(re.sub(pattern, '', str(source))).split(',')

def create_vocabulary(documents: list):
    vocabulary = []
    for sentence in tqdm(documents):
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)
    return vocabulary

def create_idx_pairs(documents: list, word2idx: dict, window_size: int = 4):
    for sentence in documents:
        indices = [word2idx[word] for word in sentence]
        for center_word_pos in range(len(indices)):
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w
                if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                    continue
                context_word_idx = indices[context_word_pos]
                yield (indices[center_word_pos], context_word_idx)

def run_model(vocabulary_size: int, documents: list, word2idx: dict, embedding_dims: int = 128, num_epochs: int = 101, learning_rate: float = 0.001):
    W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
    W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)

    for epo in range(num_epochs):
        start_time = time.time()
        loss_val = 0
        idx_pairs = create_idx_pairs(documents, word2idx)
        for data, target in idx_pairs:
            x = Variable(get_input_layer(data)).float()
            y_true = Variable(torch.from_numpy(np.array([target])).long())

            z1 = torch.matmul(W1, x)
            z2 = torch.matmul(W2, z1)
            log_softmax = F.log_softmax(z2, dim=0)

            loss = F.nll_loss(log_softmax.view(1,-1), y_true)
            loss_val += loss.data.item()
            loss.backward()

            with torch.no_grad():
                W1 -= learning_rate * W1.grad
                W2 -= learning_rate * W2.grad

                W1.grad.zero_()
                W2.grad.zero_()

        print('Loss at epo {0}: {1}; {2} seconds'.format(epo,
                                                         loss_val/len(idx_pairs),
                                                         int(time.time()-start_time)))
    return W1, W2


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    documents = list(read_input(namespace.data))
    # vocabulary = create_vocabulary(documents)
    vocabulary = list(OrderedDict.fromkeys(ha(documents)).keys())
    vocabulary_size = len(vocabulary)
    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

    W1, W2 = run_model(vocabulary_size, documents, word2idx)

    with open('vocabulary.pickle', 'wb') as f:
        pickle.dump(vocabulary, f)

    with open('w1.pickle', 'wb') as f:
        pickle.dump(W1, f)
    with open('w2.pickle', 'wb') as f:
        pickle.dump(W2, f)

    with torch.no_grad():
        embedding_w1 = {word: torch.matmul(W1, get_input_layer(i)).numpy()
                        for i, word in enumerate(vocabulary)}

    with open('embedding_w1.pickle', 'wb') as f:
        pickle.dump(embedding_w1, f)

    with torch.no_grad():
        embedding_full = {word: W1[:, i] * W2[i]
                          for i, word in enumerate(vocabulary)}

    with open('embedding_full.pickle', 'wb') as f:
        pickle.dump(embedding_full, f)
