from model.model_config import DefaultConfig
from model.embedding import Embedding
from model.loss import sequence_loss
from data_generator.vocab import Vocab
from util import constant

import tensorflow as tf
from tensor2tensor.layers import common_attention
from tensor2tensor.models import transformer

def get_graph_data(data,
                   sentence_simple_input,
                   sentence_complex_input,
                   model_config=None):
    input_feed = {}
    model_config = (DefaultConfig()
                    if model_config is None else model_config)
    voc = Vocab()

    tmp_sentence_simple, tmp_sentence_complex = [],[]
    for i in range(model_config.batch_size):
        sentence_simple, sentence_complex = data.get_data_sample()

        # PAD <s>, </s>, <pad>, <go>
        if len(sentence_simple) < model_config.max_simple_sentence:
            num_pad = model_config.max_simple_sentence - len(sentence_simple)
            sentence_simple.append(num_pad * voc.encode(constant.SYMBOL_PAD))
        # sentence_simple.insert(voc.encode(constant.SYMBOL_START), 0)
        # sentence_simple.append(voc.encode(constant.SYMBOL_END))
        if len(sentence_complex) < model_config.max_complex_sentence:
            num_pad = model_config.max_complex_sentence - len(sentence_complex)
            sentence_complex.append(num_pad * voc.encode(constant.SYMBOL_PAD))
        # sentence_complex.insert(voc.encode(constant.SYMBOL_START), 0)
        # sentence_complex.insert(voc.encode(constant.SYMBOL_GO), 0)
        # sentence_complex.append(voc.encode(constant.SYMBOL_END))

        tmp_sentence_simple.append(sentence_simple)
        tmp_sentence_complex.append(sentence_complex)

    for step in range(model_config.max_simple_sentence):
        input_feed[sentence_simple_input[step].name] = [tmp_sentence_simple[batch_idx][step]
                                                        for batch_idx in range(model_config.batch_size)]
    for step in range(model_config.max_complex_sentence):
        input_feed[sentence_complex_input[step].name] = [tmp_sentence_complex[batch_idx][step]
                                                         for batch_idx in range(model_config.batch_size)]

    return input_feed

class Graph():
    def __init__(self, data, model_config=None):
        self.model_config = (DefaultConfig()
                             if model_config is None else model_config)
        self.data = data
        self.voc = Vocab()
        self.hparams = transformer.transformer_base()

    def create_model(self):
        with tf.variable_scope("variables"):
            self.sentence_simple_input_placeholder = []
            for step in range(self.model_config.max_simple_sentence):
                self.sentence_simple_input_placeholder.append(tf.zeros(self.model_config.batch_size,
                                                           tf.int32, name='simple_input'))

            self.sentence_complex_input_placeholder = []
            for step in range(self.model_config.max_complex_sentence):
                self.sentence_complex_input_placeholder.append(tf.zeros(self.model_config.batch_size,
                                                            tf.int32, name='simple_input'))

            self.emb_simple = Embedding(self.data.vocab_simple, self.model_config).get_embedding()
            self.emb_complex = Embedding(self.data.vocab_complex, self.model_config).get_embedding()

        with tf.variable_scope("inputs"):
            self.sentence_simple_input = tf.stack(self.sentence_simple_input_placeholder, axis=1)
            self.sentence_complex_input = tf.stack(self.sentence_complex_input_placeholder, axis=1)

            simple_input = tf.nn.embedding_lookup(self.emb_simple, self.sentence_simple_input)
            complex_input = tf.nn.embedding_lookup(self.emb_complex, self.sentence_complex_input)

            simple_input_bias = common_attention.attention_bias_ignore_padding(
                tf.to_float(tf.equal(self.sentence_simple_input, self.voc.encode(constant.SYMBOL_PAD))))
            complex_input_bias = common_attention.attention_bias_ignore_padding(
                tf.to_float(tf.equal(self.sentence_complex_input, self.voc.encode(constant.SYMBOL_PAD))))

        with tf.variable_scope("transformer"):
            encoder = transformer.transformer_encoder(complex_input, complex_input_bias,
                                            self.hparams)
            decoder = transformer.transformer_decoder(simple_input, encoder,
                                            simple_input_bias, complex_input_bias,
                                            self.hparams)

        with tf.variable_scope("output"):
            initializer = tf.random_uniform_initializer(minval=-0.08, maxval=0.08)
            w = tf.get_variable('output_w',
                                shape=[1, self.model_config.dimension, len(self.voc.i2w)], initializer=initializer)
            b = tf.get_variable('output_b',
                                shape=[1,len(self.voc.i2w)], initializer=initializer)
            output = tf.nn.conv1d(decoder, w, 1, 'SAME')
            output = tf.add(output, b)

        with tf.variable_scope('optimization'):
            self.global_step = tf.get_variable('global_step',
                                          initializer=tf.constant(0, dtype=tf.int64), trainable=False)
            self.loss = sequence_loss(output, self.sentence_simple_input)
            self.train_op = self.create_train_op()
            self.increment_global_step = tf.assign_add(self.global_step, 1)
            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

    def create_train_op(self):
        if self.model_config.optimizer == 'adagrad':
            opt = tf.train.AdagradOptimizer(self.model_config.learning_rate)
        # Adam need lower learning rate
        elif self.model_config.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(self.model_config.learning_rate)
        else:
            raise Exception('Not Implemented Optimizer!')

        if self.model_config.max_grad_staleness > 0:
            opt = tf.contrib.opt.DropStaleGradientOptimizer(opt, self.model_config.max_grad_staleness)

        grads_and_vars = opt.compute_gradients(self.loss, var_list=tf.trainable_variables())
        grads = [g for (g,v) in grads_and_vars]
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.model_config.max_grad_norm)

        return opt.apply_gradients(zip(clipped_grads, tf.trainable_variables()), global_step=self.global_step)


if __name__ == '__main__':
    from data_generator.data import Data
    data = Data('../data/dummy_simple_dataset', '../data/dummy_complex_dataset',
                '../data/dummy_simple_vocab', '../data/dummy_complex_vocab')
    # get_graph_data(data, None, None)
    graph = Graph(data)
    graph.create_model()