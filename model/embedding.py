from model.model_config import DefaultConfig

import tensorflow as tf
from util import constant
import numpy as np

class Embedding:
    def __init__(self, voc_complex, voc_simple, model_config):
        self.model_config = model_config
        self.voc_complex = voc_complex
        self.voc_simple = voc_simple
        # self.init = tf.truncated_normal_initializer(
        #     stddev=self.model_config.trunc_norm_init_std)
        self.emb_init = tf.random_uniform_initializer(-0.1, 0.1)
        self.w_init = tf.contrib.layers.xavier_initializer() # tf.random_uniform_initializer(-0.08, 0.08)
        print('Use tied embedding: \t%s.' % self.model_config.tie_embedding)

    def get_complex_embedding(self):
        with tf.device('/cpu:0'):
            if hasattr(self, 'emb_complex'):
                return self.emb_complex
            self.emb_complex = tf.get_variable(
                'embedding_complex', [self.voc_complex.vocab_size(),self.model_config.dimension], tf.float32,
                initializer=self.emb_init, trainable=self.model_config.train_emb)
            if self.model_config.init_vocab_emb_complex:
                np_emb_complex = np.loadtxt(self.model_config.init_vocab_emb_complex)
                tmp = np.zeros([constant.REVERED_VOCAB_SIZE, self.model_config.dimension])
                np_emb_complex = np.concatenate((tmp, np_emb_complex), axis=0)
                np_emb_complex = np_emb_complex[:self.voc_complex.vocab_size()]
                self.vocab_embs_complex_init_fn = self.emb_complex.assign(
                    tf.convert_to_tensor(np_emb_complex, dtype=tf.float32))
                print('Init complex embs from %s' % self.model_config.init_vocab_emb_complex)
            return self.emb_complex

    def get_simple_embedding(self):
        with tf.device('/cpu:0'):
            if hasattr(self, 'emb_simple'):
                return self.emb_simple

            if (self.model_config.tie_embedding == 'none' or
                        self.model_config.tie_embedding == 'dec_out'):
                self.emb_simple = tf.get_variable(
                    'embedding_simple', [self.voc_simple.vocab_size(), self.model_config.dimension], tf.float32,
                    initializer=self.emb_init, trainable=self.model_config.train_emb)
                if self.model_config.init_vocab_emb_simple:
                    np_emb_simple = np.loadtxt(self.model_config.init_vocab_emb_simple)
                    tmp = np.zeros([constant.REVERED_VOCAB_SIZE, self.model_config.dimension])
                    np_emb_simple = np.concatenate((tmp, np_emb_simple), axis=0)
                    np_emb_simple = np_emb_simple[:self.voc_simple.vocab_size()]
                    self.vocab_embs_simple_init_fn = self.emb_simple.assign(
                        tf.convert_to_tensor(np_emb_simple, dtype=tf.float32))
                    print('Init simple embs from %s' % self.model_config.init_vocab_emb_simple)
                return self.emb_simple
            else:
                return self.get_complex_embedding()

    def get_w(self):
        if self.model_config.framework == 'transformer':
            if self.model_config.tie_embedding == 'none':
                self.proj_w = tf.get_variable(
                    'output_w', [self.voc_simple.vocab_size(), self.model_config.dimension], tf.float32,
                    initializer=self.w_init)
                return self.proj_w
            elif self.model_config.tie_embedding == 'enc_dec':
                self.proj_w = tf.get_variable(
                    'output_w', [self.voc_complex.vocab_size(), self.model_config.dimension], tf.float32,
                    initializer=self.w_init)
                return self.proj_w
            elif self.model_config.tie_embedding == 'dec_out':
                return self.get_simple_embedding()
            elif self.model_config.tie_embedding == 'all':
                return self.get_complex_embedding()
            else:
                raise NotImplementedError('Not Implemented tie_embedding option.')
        elif self.model_config.framework == 'seq2seq':
            self.proj_w = tf.get_variable(
                'output_w', [self.voc_simple.vocab_size(), self.model_config.dimension], tf.float32,
                initializer=self.w_init)
            return self.proj_w

    def get_b(self):
        if self.model_config.framework == 'transformer':
            if (self.model_config.tie_embedding == 'none' or
                        self.model_config.tie_embedding == 'enc_dec' or
                        self.model_config.tie_embedding == 'dec_out'):
                self.proj_b = tf.get_variable('output_b',
                                              shape=[self.voc_simple.vocab_size()], initializer=self.w_init)
                return self.proj_b
            elif self.model_config.tie_embedding == 'all':
                self.proj_b = tf.get_variable('output_b',
                                              shape=[self.voc_complex.vocab_size()], initializer=self.w_init)
                return self.proj_b
        elif self.model_config.framework == 'seq2seq':
            self.proj_b = tf.get_variable('output_b',
                                          shape=[self.voc_simple.vocab_size()], initializer=self.w_init)
            return self.proj_b






