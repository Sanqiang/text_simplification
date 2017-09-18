from model.embedding import Embedding
from model.loss import sequence_loss
from util import constant

import tensorflow as tf
from tensor2tensor.layers import common_attention
from tensor2tensor.models import transformer
from tensor2tensor.utils import beam_search


class Graph:
    def __init__(self, data, is_train, model_config):
        self.model_config = model_config
        self.data = data
        self.is_train = is_train
        self.hparams = transformer.transformer_base()
        self.setup_hparams()
        print('Batch Szie:\t%d.' % self.model_config.batch_size)

    def transformer_fn(self):
        def decode_step(decode_input_list):
            batch_go = [tf.zeros([self.model_config.batch_size, self.model_config.dimension])]
            target_length = len(decode_input_list) + 1
            decoder_emb_inputs = tf.stack(batch_go + decode_input_list, axis=1)
            decoder_output = self.decode_inputs_to_outputs(
                decoder_emb_inputs, encoder_outputs, encoder_attn_bias)
            decoder_output_list = [
                tf.squeeze(d, 1)
                for d in tf.split(decoder_output, target_length, axis=1)]
            return decoder_output_list

        encoder_embed_inputs = tf.stack(
            self.embedding_fn(self.sentence_complex_input_placeholder, self.emb_complex), axis=1)
        encoder_attn_bias = common_attention.attention_bias_ignore_padding(
            tf.to_float(tf.equal(tf.stack(self.sentence_complex_input_placeholder, axis=1),
                                 self.data.vocab_complex.encode(constant.SYMBOL_PAD))))
        encoder_embed_inputs = common_attention.add_timing_signal_1d(encoder_embed_inputs)

        with tf.variable_scope('transformer_encoder'):
            encoder_embed_inputs = tf.nn.dropout(encoder_embed_inputs,
                                                 1.0 - self.hparams.layer_prepostprocess_dropout)
            encoder_outputs = transformer.transformer_encoder(
                encoder_embed_inputs, encoder_attn_bias, self.hparams)

        with tf.variable_scope('transformer_decoder'):
            decoder_embed_inputs = []
            decoder_output_list = []
            decoder_target_list = []
            decoder_logit_list = []
            if not self.model_config.train_with_hyp and self.model_config.beam_search_size < 1:
                # General train
                print('Use Generally Process.')
                decoder_embed_inputs = self.embedding_fn(
                    self.sentence_simple_input_placeholder[:-1], self.emb_simple)
                decoder_output_list = decode_step(decoder_embed_inputs)
                decoder_target_list = self.sentence_simple_input_placeholder
                decoder_logit_list = [self.output_to_logit(o) for o in decoder_output_list]
            elif self.model_config.train_with_hyp or self.model_config.beam_search_size <= 1:
                # Greedy search
                print('Use Greedy Search.')
                for step in range(self.model_config.max_simple_sentence):
                    if step > 0:
                        tf.get_variable_scope().reuse_variables()

                    decoder_output_list = decode_step(decoder_embed_inputs)
                    last_logits = self.output_to_logit(decoder_output_list[-1])
                    last_outid = tf.cast(tf.argmax(last_logits, 1), tf.int32)
                    if self.is_train:
                        decoder_target_list.append(self.sentence_simple_input_placeholder[step])
                    else:
                        decoder_target_list.append(last_outid)
                    decoder_logit_list.append(last_logits)
                    decoder_embed_inputs += self.embedding_fn([last_outid], self.emb_simple)
            else:
                # Beam Search
                print('Use Beam Search with Beam Search Size %d.' % self.model_config.beam_search_size)
                return self.transformer_beam_search(encoder_outputs, encoder_attn_bias)

        return decoder_output_list, decoder_logit_list, decoder_target_list

    def transformer_beam_search(self, encoder_outputs, encoder_attn_bias):
        # Use Beam Search in evaluation stage
        # Update [a, b, c] to [a, a, a, b, b, b, c, c, c] if beam_search_size == 3
        encoder_outputs = tf.concat(
            [tf.tile(tf.expand_dims(encoder_outputs[o, :, :], axis=0),
                     [self.model_config.beam_search_size, 1, 1])
             for o in range(self.model_config.batch_size)], axis=0)

        encoder_attn_bias = tf.concat(
            [tf.tile(tf.expand_dims(encoder_attn_bias[o, :, :, :], axis=0),
                     [self.model_config.beam_search_size, 1, 1, 1])
             for o in range(self.model_config.batch_size)], axis=0)

        def symbol_to_logits_fn(ids):
            embs = tf.nn.embedding_lookup(self.emb_simple, ids[:, 1:])
            embs = tf.pad(embs, [[0, 0], [1, 0], [0, 0]])
            decoder_outputs = self.decode_inputs_to_outputs(embs, encoder_outputs, encoder_attn_bias)
            return self.output_to_logit(decoder_outputs[:, -1, :])

        beam_ids, beam_score = beam_search.beam_search(symbol_to_logits_fn,
                                                       tf.zeros([self.model_config.batch_size], tf.int32),
                                                       self.model_config.beam_search_size,
                                                       self.model_config.max_simple_sentence,
                                                       len(self.data.vocab_simple.i2w),
                                                       0.6,
                                                       self.data.vocab_simple.encode(constant.SYMBOL_END))
        top_beam_ids = beam_ids[:, 0, 1:]
        top_beam_ids = tf.pad(top_beam_ids,
                              [[0, 0],
                               [0, self.model_config.max_simple_sentence - tf.shape(top_beam_ids)[1]]])
        decoder_target_list = [tf.squeeze(d, 1)
                               for d in tf.split(top_beam_ids, self.model_config.max_simple_sentence, axis=1)]
        decoder_logit_list = -beam_score[:, 0] / tf.to_float(tf.shape(top_beam_ids)[1])
        return None, decoder_logit_list, decoder_target_list

    def embedding_fn(self, inputs, embedding):
        if not inputs:
            return []
        else:
            return [tf.nn.embedding_lookup(embedding, inp) for inp in inputs]

    def output_to_logit(self, prev_out):
        prev_logit = tf.add(tf.matmul(prev_out, self.w), self.b)
        return prev_logit

    def decode_inputs_to_outputs(self, decoder_embed_inputs, encoder_outputs, encoder_attn_bias):
        decoder_embed_inputs = common_attention.add_timing_signal_1d(decoder_embed_inputs)
        decoder_attn_bias = common_attention.attention_bias_lower_triangle(tf.shape(decoder_embed_inputs)[1])
        decoder_embed_inputs = tf.nn.dropout(decoder_embed_inputs,
                                             1.0 - self.hparams.layer_prepostprocess_dropout)
        return transformer.transformer_decoder(decoder_embed_inputs,
                                               encoder_outputs,
                                               decoder_attn_bias,
                                               encoder_attn_bias,
                                               self.hparams)

    def setup_hparams(self):
        self.hparams.hidden_size = self.model_config.dimension
        if self.is_train:
            self.hparams.add_hparam('mode', tf.estimator.ModeKeys.TRAIN)
        else:
            self.hparams.add_hparam('mode', tf.estimator.ModeKeys.EVAL)

    def create_model(self):
        with tf.variable_scope('variables'):
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

            initializer = tf.random_uniform_initializer(minval=-0.08, maxval=0.08)
            self.w = tf.get_variable('output_w',
                                     shape=[self.model_config.dimension, len(self.data.vocab_simple.i2w)], initializer=initializer)
            self.b = tf.get_variable('output_b',
                                     shape=[len(self.data.vocab_simple.i2w)], initializer=initializer)

        with tf.variable_scope('transformer'):
            decoder_output_list, decoder_logit_list, decoder_target_list = self.transformer_fn()
            self.decoder_target_list = tf.stack(decoder_target_list, axis=-1)
            if not self.is_train and self.model_config.beam_search_size > 1:
                # in beam search, decoder_logit_list is beam score
                self.loss = tf.reduce_mean(decoder_logit_list)
            else:
                decode_word_weight = [tf.to_float(tf.not_equal(d, self.data.vocab_simple.encode(constant.SYMBOL_PAD)))
                          for d in decoder_target_list]
                self.loss = sequence_loss(tf.stack(decoder_logit_list, axis=1),
                                          tf.stack(decoder_target_list, axis=1),
                                          tf.stack(decode_word_weight, axis=1))

        with tf.variable_scope('optimization'):
            self.global_step = tf.get_variable('global_step',
                                               initializer=tf.constant(0, dtype=tf.int64), trainable=False)

            if self.is_train:
                self.increment_global_step = tf.assign_add(self.global_step, 1)
                self.train_op = self.create_train_op()

            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        print('Graph Built.')

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
