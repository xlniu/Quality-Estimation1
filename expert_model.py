# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
import tensorflow as tf

from exp_hyperparams import EXP_Hyperparams as hp
from exp_data_load import *
from modules import *
from modules import _get_embed_device
import os, codecs
import os
import time

class BilingualExpert():
    def __init__(self, is_training=True):

        self.x = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.maxlen))
        self.y = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.maxlen))

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
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget)/ (tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)

        if is_training:
            # Loss
            self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=hp.vocab_size))
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
            self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))

            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.learning_rate = tf.constant(hp.lr)
            self.learning_rate = self._get_learning_rate_warmup()
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.998, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

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

def create_or_load_model(model, session, name):
    latest_ckpt = tf.train.latest_checkpoint(hp.log_dir)
    if latest_ckpt:
        model = load_model(model, latest_ckpt, session, name)
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print("  created %s model with fresh parameters, time %.2fs" % (name, time.time() - start_time))
    global_step = model.global_step.eval(session=session)
    return model, global_step

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    # Prepare data
    data_set = load_train_data(hp.source_train, hp.target_train, "train")
    # Construct graph
    model = BilingualExpert("train"); print("Graph loaded")

    # num = 0
    # for var in tf.trainable_variables():
    #     print(var.name,var.shape)
    #     temp = 1
    #     for i in var.shape:
    #         temp *= int(i)
    #     num += temp
    # print(num*8)
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
        # This is the training loop.
        while global_step < hp.num_train_steps:
            # Run a step
            x, y = get_batch_data(data_set)
            sess.run(model.train_op,feed_dict={
                        loaded_train_model.x: x,
                        loaded_train_model.y: y,
                    })
            global_step = sess.run(model.global_step)
            # Once in a while, we print statistics.
            if global_step - last_stats_step >= hp.steps_per_stats:
                last_stats_step = global_step
                mean_loss = sess.run(model.mean_loss,feed_dict={
                        loaded_train_model.x: x,
                        loaded_train_model.y: y,
                    })
                print("global_step : %d, sent_loss : %f, time %.2fs"%(global_step,mean_loss,time.time()-start_time))
                start_time = time.time()

            if global_step - last_save_step >= hp.steps_per_save:
                last_save_step = global_step

                # Save checkpoint
                loaded_train_model.saver.save(
                    sess,
                    os.path.join(hp.log_dir, "exp.ckpt"),
                    global_step=global_step)
                print("# Save, global step %d" % global_step)

        # Done training
        model.saver.save(
            sess,
            os.path.join(hp.log_dir, "exp.ckpt"),
            global_step=global_step)

        coord.request_stop()
        coord.join(threads)

        print("# Done training!" )
    

