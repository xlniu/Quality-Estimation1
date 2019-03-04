# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
import tensorflow as tf

from qe_hyperparams import QE_Hyperparams as hp
from qe_data_load import *
from modules import *
from modules import _get_embed_device
import os, codecs
from tqdm import tqdm
import os
import time
import math

class QEModel():
    def __init__(self, is_training=True):

        self.x = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.maxlen))
        self.y = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.maxlen))
        self.label = tf.placeholder(tf.float32, shape=(hp.batch_size, ))

        # define decoder inputs
        self.decoder_forward_inputs = self.y
        self.decoder_backward_inputs = self.y[:, ::-1] # 逆序

        # Encoder
        with tf.variable_scope("encoder"):
            ## Embedding
            self.enc = embedding(self.x,
                                  vocab_size=hp.vocab_size,
                                  num_units=hp.hidden_units,
                                  scale=True,
                                  scope="enc_embed")

            ## Positional Encoding
            if hp.sinusoid:
                self.enc += positional_encoding(self.x,
                                  num_units=hp.hidden_units,
                                  zero_pad=False,
                                  scale=False,
                                  scope="enc_pe")
            else:
                self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                                  vocab_size=hp.maxlen,
                                  num_units=hp.hidden_units,
                                  zero_pad=False,
                                  scale=False,
                                  scope="enc_pe")


            ## Dropout
            self.enc = tf.layers.dropout(self.enc,
                                        rate=hp.dropout_rate,
                                        training=tf.convert_to_tensor(is_training))

            ## Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.enc = multihead_attention(queries=self.enc,
                                                    keys=self.enc,
                                                    num_units=hp.hidden_units,
                                                    num_heads=hp.num_heads,
                                                    dropout_rate=hp.dropout_rate,
                                                    is_training=is_training,
                                                    causality=False)

                    ### Feed Forward
                    self.enc = feedforward(self.enc, num_units=[4*hp.hidden_units, hp.hidden_units])

        with tf.variable_scope("decoder"):
            # forward Decoder
            with tf.variable_scope("forward_decoder"):
                ## Embedding
                self.forward_dec = embedding(self.decoder_forward_inputs,
                                      vocab_size=hp.vocab_size,
                                      num_units=hp.hidden_units,
                                      scale=True,
                                      scope="dec_embed")
                self.fw_decoder_emb_inp = self.forward_dec # 后面会用到

                ## Positional Encoding
                if hp.sinusoid:
                    self.forward_dec += positional_encoding(self.decoder_forward_inputs,
                                      num_units=hp.hidden_units,
                                      zero_pad=False,
                                      scale=False,
                                      scope="dec_pe")
                else:
                    self.forward_dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_forward_inputs)[1]), 0), [tf.shape(self.decoder_forward_inputs)[0], 1]),
                                      vocab_size=hp.maxlen,
                                      num_units=hp.hidden_units,
                                      zero_pad=False,
                                      scale=False,
                                      scope="dec_pe")

                ## Dropout
                self.forward_dec = tf.layers.dropout(self.forward_dec,
                                            rate=hp.dropout_rate,
                                            training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):

                        ## Multihead Attention ( self-attention)
                        self.forward_dec = multihead_attention(queries=self.forward_dec,
                                                        keys=self.forward_dec,
                                                        num_units=hp.hidden_units,
                                                        num_heads=hp.num_heads,
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=True,
                                                        scope="self_attention")

                        ## Multihead Attention ( vanilla attention)
                        self.forward_dec = multihead_attention(queries=self.forward_dec,
                                                        keys=self.enc,
                                                        num_units=hp.hidden_units,
                                                        num_heads=hp.num_heads,
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=False,
                                                        scope="vanilla_attention")

                        ## Feed Forward
                        self.forward_dec = feedforward(self.forward_dec, num_units=[4*hp.hidden_units, hp.hidden_units])

            # backward Decoder
            with tf.variable_scope("backward_decoder"):
                ## Embedding
                self.backward_dec = embedding(self.decoder_backward_inputs,
                                      vocab_size=hp.vocab_size,
                                      num_units=hp.hidden_units,
                                      scale=True,
                                      scope="dec_embed")


                ## Positional Encoding
                if hp.sinusoid:
                    self.backward_dec += positional_encoding(self.decoder_backward_inputs,
                                      num_units=hp.hidden_units,
                                      zero_pad=False,
                                      scale=False,
                                      scope="dec_pe")
                else:
                    self.backward_dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_backward_inputs)[1]), 0), [tf.shape(self.decoder_backward_inputs)[0], 1]),
                                      vocab_size=hp.maxlen,
                                      num_units=hp.hidden_units,
                                      zero_pad=False,
                                      scale=False,
                                      scope="dec_pe")

                ## Dropout
                self.backward_dec = tf.layers.dropout(self.backward_dec,
                                            rate=hp.dropout_rate,
                                            training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):

                        ## Multihead Attention ( self-attention)
                        self.backward_dec = multihead_attention(queries=self.backward_dec,
                                                        keys=self.backward_dec,
                                                        num_units=hp.hidden_units,
                                                        num_heads=hp.num_heads,
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=True,
                                                        scope="self_attention")

                        ## Multihead Attention ( vanilla attention)
                        self.backward_dec = multihead_attention(queries=self.backward_dec,
                                                        keys=self.enc,
                                                        num_units=hp.hidden_units,
                                                        num_heads=hp.num_heads,
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=False,
                                                        scope="vanilla_attention")

                        ## Feed Forward
                        self.backward_dec = feedforward(self.backward_dec, num_units=[4*hp.hidden_units, hp.hidden_units])

                self.backward_dec_rev = self.backward_dec[:,::-1,:] # 注意

            shift_outputs = shift_concat(
                (self.forward_dec, self.backward_dec_rev),
                None)

            shift_inputs = shift_concat(
                (self.fw_decoder_emb_inp, self.fw_decoder_emb_inp),
                None)


            shift_proj_inputs = tf.layers.dense(shift_inputs, 2 * hp.hidden_units, use_bias=False, name="emb_proj_layer")
            _pre_qefv = tf.concat([shift_outputs, shift_proj_inputs], axis=-1)

            # Final linear projection
            self.logits = tf.layers.dense(_pre_qefv, hp.vocab_size, use_bias=False, name="output_projection") # batch*seq*N
            self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))

            # Extract logits feature for mismatching
            shape = tf.shape(self.decoder_forward_inputs) # batch*seq
            idx0 = tf.expand_dims(tf.range(shape[0]), -1) # batch*1
            idx0 = tf.tile(idx0, [1, shape[1]]) # batch*seq
            idx0 = tf.cast(idx0, self.decoder_forward_inputs.dtype)
            idx1 = tf.expand_dims(tf.range(shape[1]), 0) # 1*seq
            idx1 = tf.tile(idx1, [shape[0], 1]) # batch*seq
            idx1 = tf.cast(idx1, self.decoder_forward_inputs.dtype)
            indices_real = tf.stack([idx0, idx1, self.decoder_forward_inputs], axis=-1)
            logits_mt = tf.gather_nd(self.logits, indices_real)
            logits_max = tf.reduce_max(self.logits, axis=-1)
            logits_diff = tf.subtract(logits_max, logits_mt)
            logits_same = tf.cast(tf.equal(self.preds, self.decoder_forward_inputs), tf.float32)
            logits_fea = tf.stack([logits_mt, logits_max, logits_diff, logits_same], axis=-1)

            # Extract QEFV
            with tf.variable_scope("output_projection", reuse=True):
                output_layer_weights = tf.get_variable('kernel') # 2048*num_target_words,这个矩阵中的每个向量类似于一个门，决定要不要取对应词
            _w = tf.nn.embedding_lookup(tf.transpose(output_layer_weights), self.decoder_forward_inputs) # batch*seq*2048
            pre_qefv = _w * _pre_qefv # batch*seq*2048,为什么要这样做？？？
            post_qefv = tf.concat([self.forward_dec, self.backward_dec_rev], axis=-1) # 注意和前面shift_concat的区别
            qefv = tf.concat([pre_qefv, post_qefv], axis=-1) # 3072

            expert_fea = tf.concat([qefv, logits_fea], axis=-1)

        with tf.variable_scope("estimator") as scope:
            estimator_outputs, estimator_states = bidirectional_lstm_encoder(expert_fea,
                                                        num_units=hp.hidden_units,
                                                        scope="bi_lstm")
            output_state_fw, output_state_bw = estimator_states
            sent_fea = tf.concat([output_state_fw[1], output_state_bw[1]], axis=-1)
            sent_logits = tf.layers.dense(sent_fea, 1)
            sent_logits = tf.squeeze(sent_logits)

        if is_training:
            # Loss
            self.sent_pred = tf.sigmoid(sent_logits)
            self.sent_loss = tf.reduce_mean(tf.square(self.label - self.sent_pred))

            self.params = tf.trainable_variables()
            self.exp_params = [var for var in self.params if 'encoder' in var.name or 'decoder' in var.name]
            self.est_params = [var for var in self.params if 'estimator' in var.name]

            # define actual trainable params
            params = self.est_params if hp.fixed_exp else self.params

            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.learning_rate = tf.constant(hp.lr)
            self.learning_rate = self._get_learning_rate_warmup()
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.998, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.sent_loss, global_step=self.global_step, var_list=params)

            # Saver
            self.saver = tf.train.Saver(
                tf.global_variables(), max_to_keep=hp.num_keep_ckpts)

    def _get_learning_rate_warmup(self):
        warmup_steps = hp.warmup_steps
        print("  learning_rate=%g, warmup_steps=%d" % (hp.lr, warmup_steps))

        step_num = tf.to_float(self.global_step) / 2. + 1
        inv_decay = hp.hidden_units ** -0.5 * tf.minimum(step_num * warmup_steps ** -1.5, step_num ** -0.5)
        return inv_decay * self.learning_rate

def load_model(model, ckpt, session, name):
    start_time = time.time()
    model.saver.restore(session, ckpt)
    session.run(tf.tables_initializer())
    print("  loaded %s model parameters from %s, time %.2fs" % (name, ckpt, time.time() - start_time))
    return model

def load_expert_weights(exp_model_dir, model, session):
    """
    call this function after session.run(tf.global_variables_initializer())
    """
    checkpoint_state = tf.train.get_checkpoint_state(exp_model_dir)
    if not checkpoint_state:
        print("# No checkpoint file found in directory: %s" % exp_model_dir)
        return

    checkpoint = checkpoint_state.all_model_checkpoint_paths[-1]
    reader = tf.train.NewCheckpointReader(checkpoint)
    var_to_shape_map = reader.get_variable_to_shape_map()

    var_values = {}
    for name in var_to_shape_map:
        if name != "global_step" and not "OptimizeLoss" in name:
            print("  loading weights of variable %s" % (name))
            var_values[name+":0"] = reader.get_tensor(name)

    loaded_vars = [var for var in model.exp_params if var.name in var_values]
    placeholders = [tf.placeholder(var.dtype, shape=var.shape) for var in loaded_vars]
    assign_ops = [tf.assign(v, p) for (v, p) in zip(loaded_vars, placeholders)]
    for p, assign_op, var in zip(placeholders, assign_ops, loaded_vars):
        session.run(assign_op, {p: var_values[var.name]})

def create_or_load_model(model, session, name):
    latest_ckpt = tf.train.latest_checkpoint(hp.model_dir)
    if latest_ckpt:
        model = load_model(model, latest_ckpt, session, name)
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print("  created %s model with fresh parameters, time %.2fs" % (name, time.time() - start_time))
    global_step = model.global_step.eval(session=session)
    if global_step == 0:
        if hp.log_dir:
            start_time = time.time()
            load_expert_weights(hp.log_dir, model, session)
            print("  load pretrained expert weights for %s model, time %.2fs" % (name, time.time() - start_time))
    return model, global_step
def evaluate(data_set, sess, model):
    pred = np.array([])
    ref = np.array([])
    X, Y, Labels = [], [], []
    for x, y, label in data_set:
        X.append(x)
        Y.append(y)
        Labels.append(label)
        if len(X) == hp.batch_size:
            X, Y, Labels = padding(X,Y,Labels)
            sent_pred = sess.run(model.sent_pred,feed_dict={
            model.x: X,
            model.y: Y,
            })
            pred = np.concatenate((pred,sent_pred))
            ref = np.concatenate((ref,Labels))
            X, Y, Labels = [], [], []
    # 不足一个batch的样本
    remain_num = len(X)
    if remain_num>0:
        while len(X)<hp.batch_size:
            X.append(X[0])
            Y.append(Y[0])
            Labels.append(Labels[0])
        X, Y, Labels = padding(X,Y,Labels)
        sent_pred = sess.run(model.sent_pred,feed_dict={
                model.x: X,
                model.y: Y,
                })
        pred = np.concatenate((pred,sent_pred[:remain_num]))
        ref = np.concatenate((ref,Labels[:remain_num]))
    pearson = np.corrcoef(pred, ref)[0, 1]
    return pearson

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    # Prepare data
    train_data_set = load_data(hp.source_train, hp.target_train, hp.label_train, "train")
    dev_data_set = load_data(hp.source_dev, hp.target_dev, hp.label_dev, "dev")
    test_data_set = load_data(hp.source_test, hp.target_test, hp.label_test, "test")

    # Construct graph
    model = QEModel("train"); print("Graph loaded")

    # for var in tf.trainable_variables():
    #     print(var.name)
    # print('\n end')

    config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config_proto.gpu_options.allow_growth = True

    with tf.Session(config=config_proto) as sess:
        loaded_train_model, global_step = create_or_load_model(
            model, sess, "train")

        coord = tf.train.Coordinator() # tf.train.shuffle_batch
        threads = tf.train.start_queue_runners(coord=coord)

        last_stats_step = global_step
        last_save_step = global_step
        start_time = time.time()
        best_pearson = 0
        wait = 0
        # This is the training loop.
        while global_step < hp.num_train_steps:
            # Run a step
            X, Y, Labels = get_batch_data(train_data_set)
            sess.run(loaded_train_model.train_op,feed_dict={
                        loaded_train_model.x: X,
                        loaded_train_model.y: Y,
                        loaded_train_model.label:Labels,
                    })
            global_step = sess.run(loaded_train_model.global_step)

            # Once in a while, we print statistics.
            if global_step - last_stats_step >= hp.steps_per_stats:
                last_stats_step = global_step
                sent_loss = sess.run(loaded_train_model.sent_loss,feed_dict={
                        loaded_train_model.x: X,
                        loaded_train_model.y: Y,
                        loaded_train_model.label:Labels,
                    })
                print("global_step : %d, sent_loss : %f, time %.2fs"%(global_step,sent_loss,time.time()-start_time))
                start_time = time.time()

            if global_step - last_save_step >= hp.steps_per_save:
                last_save_step = global_step
                dev_pearson = evaluate(dev_data_set, sess, loaded_train_model)
                test_pearson = evaluate(test_data_set, sess, loaded_train_model)
                print("# dev pearson : %s, test pearson : %s" % (dev_pearson, test_pearson))
                if dev_pearson > best_pearson:
                    wait = 0
                    best_pearson = dev_pearson
                    # Save checkpoint
                    loaded_train_model.saver.save(
                        sess,
                        os.path.join(hp.model_dir, "qe.ckpt"),
                        global_step=global_step)
                    print("# Save, global step %d" % (global_step))
                else:
                    wait += 1
                    if wait >= hp.patience:
                        print("Early Stop !")
                        break

        coord.request_stop()
        coord.join(threads)

        print("# Done training!" )
    

