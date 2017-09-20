from model.model_config import DefaultConfig

import tensorflow as tf
import math

class Embedding:
    def __init__(self, voc_complex, voc_simple, model_config):
        self.model_config = model_config
        self.voc_complex = voc_complex
        self.voc_simple = voc_simple
        # self.init = tf.truncated_normal_initializer(
        #     stddev=self.model_config.trunc_norm_init_std)
        self.emb_init = tf.random_uniform_initializer(-0.08, 0.08)
        self.w_init = tf.random_uniform_initializer(-0.08, 0.08)
        print('Use tie embedding: \t%s.' % self.model_config.tie_embedding)


    def get_complex_embedding(self):
        if hasattr(self, 'emb_complex'):
            return self.emb_complex

        self.emb_complex = tf.get_variable(
            'embedding_complex', [len(self.voc_complex.i2w),self.model_config.dimension], tf.float32,
            initializer=self.emb_init)
        return self.emb_complex

    def get_simple_embedding(self):
        if hasattr(self, 'emb_simple'):
            return self.emb_simple

        if (self.model_config.tie_embedding == 'none' or
                    self.model_config.tie_embedding == 'dec_out'):
            self.emb_simple = tf.get_variable(
                'embedding_simple', [len(self.voc_simple.i2w), self.model_config.dimension], tf.float32,
                initializer=self.emb_init)
            return self.emb_simple
        else:
            return self.get_complex_embedding()

    def get_w(self):
        if self.model_config.tie_embedding == 'none':
            self.proj_w = tf.get_variable(
                'output_w', [len(self.voc_simple.i2w), self.model_config.dimension], tf.float32,
                initializer=self.w_init)
            return self.proj_w
        elif self.model_config.tie_embedding == 'enc_dec':
            self.proj_w = tf.get_variable(
                'output_w', [len(self.voc_complex.i2w), self.model_config.dimension], tf.float32,
                initializer=self.w_init)
            return self.proj_w
        elif self.model_config.tie_embedding == 'dec_out':
            return self.get_simple_embedding()
        elif self.model_config.tie_embedding == 'all':
            return self.get_complex_embedding()
        else:
            raise NotImplementedError('Not Implemented tie_embedding option.')

    def get_b(self):
        if (self.model_config.tie_embedding == 'none' or
                    self.model_config.tie_embedding == 'enc_dec' or
                    self.model_config.tie_embedding == 'dec_out'):
            self.proj_b = tf.get_variable('output_b',
                                          shape=[len(self.voc_simple.i2w)], initializer=self.w_init)
            return self.proj_b
        elif self.model_config.tie_embedding == 'all':
            self.proj_b = tf.get_variable('output_b',
                                          shape=[len(self.voc_complex.i2w)], initializer=self.w_init)
            return self.proj_b





