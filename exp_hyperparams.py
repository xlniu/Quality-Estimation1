# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
maxlen = 70, batch_size = 128
源端、目标端、proj 词表共享 ？？？qe-brain中没有共享。proj词表也没有办法共享，2048*N
优化策略：已经更改为和qe-brain中相同
'''
class EXP_Hyperparams:
    '''Hyperparameters'''
    # data

    source_train = '../parallel_data/2017/tok.lower.filter.de'
    target_train = '../parallel_data/2017/tok.lower.filter.en'
    #source_test = '../parallel_data/2017/qe_data/sentence_level_en_de/dev.tok.lower.src'
    #target_test = '../parallel_data/2017/qe_data/sentence_level_en_de/dev.tok.lower.pe'
    vocab_dir = './preprocessed_qe/'
    pattern = 'de-en'
    '''
    source_train = 'corpora/train.tags.de-en.de'
    target_train = 'corpora/train.tags.de-en.en'
    source_test = 'corpora/IWSLT16.TED.tst2014.de-en.de.xml'
    target_test = 'corpora/IWSLT16.TED.tst2014.de-en.en.xml'
    vocab_dir = './preprocessed/'
    pattern = 'de-en'
    '''

    # training
    batch_size = 128 # alias = N
    lr = 2.1 # learning rate. learning rate is adjusted to the global step.
    warmup_steps = 8000
    log_dir = 'logdir_de_en_2018' # log directory
    num_keep_ckpts = 5
    steps_per_stats = 10 # Once in a while, we print statistics.
    steps_per_save = 10000

    # model
    maxlen = 70+2 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    #min_cnt = 20 # words whose occurred less than min_cnt are encoded as <UNK>.
    vocab_size = 30000 # src and tgt
    hidden_units = 512 # alias = C
    num_blocks = 2 # number of encoder/decoder blocks
    num_train_steps = 5000000
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = True # If True, use sinusoid. If false, positional embedding.
    
    
    
    
