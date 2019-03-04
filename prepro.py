# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
from exp_hyperparams import EXP_Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import os
import regex
from collections import Counter

vocab_size = 120000
def make_vocab(fpath, fname):
    '''Constructs vocabulary.
    
    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.
    
    Writes vocabulary line by line to `preprocessed/fname`
    '''  
    text = codecs.open(fpath, 'r', 'utf-8').read()
    text = regex.sub("[^\s\p{Latin}']", "", text)
    words = text.split()
    word2cnt = Counter(words)
    if not os.path.exists(hp.vocab_dir): os.mkdir(hp.vocab_dir)
    with codecs.open('preprocessed_qe/{}'.format(fname), 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(vocab_size):
            fout.write(u"{}\n".format(word))
            #fout.write(u"{}\t{}\n".format(word, cnt))

if __name__ == '__main__':

    make_vocab(hp.source_train, hp.pattern.split('-')[0]+".vocab.tsv")
    make_vocab(hp.target_train, hp.pattern.split('-')[1]+".vocab.tsv")
    print("Done")
