from model.model_config import DefaultConfig
from model.embedding import Embedding
from model.loss import sequence_loss
from data_generator.vocab import Vocab
from util import constant

import tensorflow as tf
from tensor2tensor.layers import common_attention
from tensor2tensor.models import transformer
from tensor2tensor.utils import beam_search

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
            sentence_simple.extend(num_pad * [voc.encode(constant.SYMBOL_PAD)])
        else:
            sentence_simple = sentence_simple[:model_config.max_simple_sentence]

        if len(sentence_complex) < model_config.max_complex_sentence:
            num_pad = model_config.max_complex_sentence - len(sentence_complex)
            sentence_complex.extend(num_pad * [voc.encode(constant.SYMBOL_PAD)])
        else:
            sentence_complex = sentence_complex[:model_config.max_complex_sentence]

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
    def __init__(self, data, is_train, model_config=None):
        self.model_config = (DefaultConfig()
                             if model_config is None else model_config)
        self.data = data
        self.is_train = is_train
        self.hparams = transformer.transformer_base()

    def encoder(self, complex_input, complex_input_bias):
        return transformer.transformer_encoder(complex_input,
                                               complex_input_bias,
                                               self.hparams)

    def decoder(self, simple_input, encoder_output, simple_input_bias, complex_input_bias):
        return transformer.transformer_decoder(simple_input,
                                               encoder_output,
                                               simple_input_bias,
                                               complex_input_bias,
                                               self.hparams)

    def output(self, decoder):
        with tf.variable_scope("output"):
            initializer = tf.random_uniform_initializer(minval=-0.08, maxval=0.08)
            w = tf.get_variable('output_w',
                                shape=[1, self.model_config.dimension, len(self.data.vocab_simple.i2w)], initializer=initializer)
            b = tf.get_variable('output_b',
                                shape=[1,len(self.data.vocab_simple.i2w)], initializer=initializer)
            output = tf.nn.conv1d(decoder, w, 1, 'SAME')
            output = tf.add(output, b)
        return output

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
            # add <go> in the beginning of decoder
            self.sentence_simple_input = tf.stack(self.sentence_simple_input_placeholder, axis=1)
            simple_go = tf.reshape(tf.stack([self.data.vocab_simple.encode(constant.SYMBOL_GO)
                                             for _ in range(self.model_config.batch_size)], axis=0),
                                   [self.model_config.batch_size, -1])
            self.sentence_simple_input = tf.concat([simple_go, self.sentence_simple_input], axis=1)

            self.sentence_complex_input = tf.stack(self.sentence_complex_input_placeholder, axis=1)

            simple_input = tf.nn.embedding_lookup(self.emb_simple, self.sentence_simple_input)
            complex_input = tf.nn.embedding_lookup(self.emb_complex, self.sentence_complex_input)
            simple_input = common_attention.add_timing_signal_1d(simple_input)
            complex_input = common_attention.add_timing_signal_1d(complex_input)

            simple_input_bias = common_attention.attention_bias_lower_triangle(tf.shape(simple_input)[1])
            # simple_input_bias = common_attention.attention_bias_ignore_padding(
            #     tf.to_float(tf.equal(self.sentence_simple_input, self.data.vocab_simple.encode(constant.SYMBOL_PAD))))
            complex_input_bias = common_attention.attention_bias_ignore_padding(
                tf.to_float(tf.equal(self.sentence_complex_input, self.data.vocab_complex.encode(constant.SYMBOL_PAD))))

        with tf.variable_scope("transformer"):
            encoder = self.encoder(complex_input, complex_input_bias)

            if self.is_train:
                decoder = self.decoder(simple_input, encoder, simple_input_bias, complex_input_bias)
                output = self.output(decoder)
                with tf.variable_scope('optimization'):
                    self.global_step = tf.get_variable('global_step',
                                                       initializer=tf.constant(0, dtype=tf.int64), trainable=False)
                    self.increment_global_step = tf.assign_add(self.global_step, 1)
                    self.loss = sequence_loss(output, self.sentence_simple_input)
                    self.train_op = self.create_train_op()

            else:
                if self.model_config.beam_search_size <= 0:
                    decoder = self.decoder(simple_input, encoder, simple_input_bias, complex_input_bias)
                    output = self.output(decoder)
                    self.target = tf.argmax(output, axis=-1)
                    self.loss = sequence_loss(output, self.sentence_simple_input)
                else:
                    # Use Beam Search in evaluation stage
                    # Update [a, b, c] to [a, a, a, b, b, b, c, c, c] if beam_search_size == 3
                    for i, encoder_output in enumerate(encoder):
                        encoder[i] = tf.concat(
                            [tf.tile(tf.expand_dims(encoder[o, :, :], axis=0),
                                     [self.model_config.beam_search_size, 1, 1])
                             for o in range(self.model_config.batch_size)], axis=0)

                    for i, encoder_bias in enumerate(complex_input_bias):
                        encoder_bias[i] = tf.concat(
                            [tf.tile(tf.expand_dims(encoder_bias[o, :, :, :], axis=0),
                                     [self.model_config.beam_search_size, 1, 1, 1])
                             for o in range(self.model_config.batch_size)], axis=0)

                    def beam_search_fn(ids):
                        embs = tf.nn.embedding_lookup(self.emb_simple, ids[:, 1:])
                        embs = tf.pad(embs, [[0, 0], [1, 0], [0, 0]], constant_values=0)
                        embs_bias = common_attention.attention_bias_lower_triangle(tf.shape(embs)[1])
                        decoder = self.decoder(embs, encoder, embs_bias, encoder_bias)
                        return decoder

                    beam_ids, beam_score = beam_search.beam_search(beam_search_fn,
                                                                   tf.zeros([self.model_config.batch_size], tf.int32),
                                                                   self.model_config.batch_size,
                                                                   self.model_config.max_simple_sentence,
                                                                   len(self.data.vocab_simple.i2w),
                                                                   0.6,
                                                                   self.data.vocab_simple.encode(constant.SYMBOL_END))
                    top_beam_ids = beam_ids[:, 0, 1:]
                    top_beam_ids = tf.pad(top_beam_ids,
                                          [[0, 0],
                                           [0, self.model_config.max_simple_sentence - tf.shape(top_beam_ids)[1]]])
                    self.target = [tf.squeeze(d, 1)
                                   for d in tf.split(top_beam_ids, self.model_config.max_simple_sentence, axis=1)]
                    self.loss = -beam_score[:, 0] / tf.to_float(tf.shape(top_beam_ids)[1])

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