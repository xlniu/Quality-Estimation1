# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
maxlen = 70, batch_size = 128
源端、目标端、proj 词表共享 ？？？qe-brain中没有共享
优化策略：已经更改为和qe-brain中相同
'''
class QE_Hyperparams:
    '''Hyperparameters'''
    # data
    source_train = '../parallel_data/2017/qe_data/sentence_level_en_de/train.tok.lower.src'
    target_train = '../parallel_data/2017/qe_data/sentence_level_en_de/train.tok.lower.mt'
    label_train = '../parallel_data/2017/qe_data/sentence_level_en_de/train.hter'
    source_dev = '../parallel_data/2017/qe_data/sentence_level_en_de/dev.tok.lower.src'
    target_dev = '../parallel_data/2017/qe_data/sentence_level_en_de/dev.tok.lower.mt'
    label_dev = '../parallel_data/2017/qe_data/sentence_level_en_de/dev.hter'
    source_test = '../parallel_data/2017/qe_data/sentence_level_en_de_test/test.2017.tok.lower.src'
    target_test = '../parallel_data/2017/qe_data/sentence_level_en_de_test/test.2017.tok.lower.mt'
    label_test = '../parallel_data/2017/qe_data/sentence_level_en_de_test/en-de_task1_test.2017.hter'

    vocab_dir = './preprocessed_qe/'
    pattern = 'en-de'

    # training
    batch_size = 64 # alias = N
    lr = 2.0 # learning rate. learning rate is adjusted to the global step.
    warmup_steps = 8000
    log_dir = 'logdir' # log directory,save expert_model
    num_keep_ckpts = 5

    # model
    maxlen = 70+2 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    #min_cnt = 20 # words whose occurred less than min_cnt are encoded as <UNK>.
    vocab_size = 30000 # src and tgt
    hidden_units = 512 # alias = C
    num_blocks = 2 # number of encoder/decoder blocks
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = True # If True, use sinusoid. If false, positional embedding.

    # qe params
    model_dir = './modeldir/' # dir of qe_model
    num_train_steps = 75000
    steps_per_stats = 10 # Once in a while, we print statistics.
    steps_per_save = 50
    fixed_exp = False # fixed expert weights or not
    patience = 5


    
    
    
    
