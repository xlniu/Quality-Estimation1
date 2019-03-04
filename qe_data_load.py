# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
from qe_hyperparams import QE_Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import random

def load_vocab(lang):
    #vocab = [line.split()[0] for line in codecs.open('preprocessed/de.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    vocab = [line.split()[0] for line in codecs.open(hp.vocab_dir + lang + '.vocab.tsv', 'r', 'utf-8').read().splitlines()]
    vocab = vocab[:hp.vocab_size]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(source_sents, target_sents, labels):
    s_word2idx, s_idx2word = load_vocab(hp.pattern.split('-')[0]) # source
    t_word2idx, t_idx2word = load_vocab(hp.pattern.split('-')[1]) # target

    # Index
    data_set = []
    for source_sent, target_sent, label in zip(source_sents, target_sents, labels):
        x = [s_word2idx.get(word, 1) for word in (u"<S>" + source_sent + u" </S>").split()] # 1: OOV, </S>: End of Text
        y = [t_word2idx.get(word, 1) for word in (u"<S>" + target_sent + u" </S>").split()]
        if max(len(x), len(y)) <=hp.maxlen:
            data_set.append([x,y,float(label)])

    return data_set

def load_data(source_file, target_file, label_file, name):
    source_sents = [line for line in codecs.open(source_file, 'r', 'utf-8').read().split("\n") if line]
    target_sents = [line for line in codecs.open(target_file, 'r', 'utf-8').read().split("\n") if line]
    labels = [line for line in codecs.open(label_file, 'r', 'utf-8').read().split("\n") if line]
    print("load %s data over. source sents : %d, target sents : %d, label : %d"%(name, len(source_sents),len(target_sents),len(labels)))
    
    data_set = create_data(source_sents, target_sents, labels) # word2id, <S></S>
    print("word2id, <S></S>")

    return data_set

def padding(x_list,y_list, label_list):
    # Pad
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    Labels = np.array(label_list)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
    return X, Y, Labels

def get_batch_data(data_set):
    X, Y, Labels = [], [], []
    for _ in range(hp.batch_size):
        x, y, label = random.choice(data_set)
        X.append(x)
        Y.append(y)
        Labels.append(label)
    X, Y, Labels = padding(X,Y,Labels)
    return X, Y, Labels

def get_batch_data_test(data_set):
    X, Y, Labels = [], [], []
    for _ in range(hp.batch_size):
        x, y, label = random.choice(data_set)
        X.append(x)
        Y.append(y)
        Labels.append(label)
    X, Y, Labels = padding(X,Y,Labels)
    return X, Y, Labels
