import tensorflow as tf
from tensor2tensor.layers import common_attention
from tensor2tensor.models import transformer
from tensor2tensor.utils import beam_search

from util import constant
from model.graph import Graph
from model.graph import ModelOutput


class TransformerGraph(Graph):
    def __init__(self, data, is_train, model_config):
        super(TransformerGraph, self).__init__(data, is_train, model_config)
        self.hparams = transformer.transformer_base()
        self.setup_hparams()
        self.model_fn = self.transformer_fn

    def transformer_fn(self):
        encoder_embed_inputs = tf.stack(
            self.embedding_fn(self.sentence_complex_input_placeholder, self.emb_complex), axis=1)
        encoder_attn_bias = common_attention.attention_bias_ignore_padding(
            tf.to_float(tf.equal(tf.stack(self.sentence_complex_input_placeholder, axis=1),
                                 self.data.vocab_complex.encode(constant.SYMBOL_PAD))))
        if self.hparams.pos == 'timing':
            encoder_embed_inputs = common_attention.add_timing_signal_1d(encoder_embed_inputs)
            print('Use positional encoding in encoder text.')

        with tf.variable_scope('transformer_encoder'):
            encoder_embed_inputs = tf.nn.dropout(encoder_embed_inputs,
                                                 1.0 - self.hparams.layer_prepostprocess_dropout)
            if self.model_config.trans_layer_gate:
                encoder_outputs = transformer.transformer_encoder_gate(
                    encoder_embed_inputs, encoder_attn_bias, self.hparams)
            else:
                encoder_outputs = transformer.transformer_encoder(
                    encoder_embed_inputs, encoder_attn_bias, self.hparams)

        with tf.variable_scope('transformer_decoder'):
            if self.is_train:
                # General train
                print('Use Generally Process.')
                decoder_embed_inputs = self.embedding_fn(
                    self.sentence_simple_input_placeholder[:-1], self.emb_simple)
                decoder_output_list = self.decode_step(
                    decoder_embed_inputs, encoder_outputs, encoder_attn_bias)
                gt_target_list = self.sentence_simple_input_placeholder
                decoder_logit_list = [self.output_to_logit(o) for o in decoder_output_list]
            else:
                # Beam Search
                print('Use Beam Search with Beam Search Size %d.' % self.model_config.beam_search_size)
                return self.transformer_beam_search(encoder_outputs, encoder_attn_bias)

        output = ModelOutput(
            decoder_outputs=decoder_output_list,
            decoder_logit_list=decoder_logit_list,
            gt_target_list=gt_target_list)
        return output

    def decode_step(self, decode_input_list, encoder_outputs, encoder_attn_bias):
        batch_go = [tf.zeros([self.model_config.batch_size, self.model_config.dimension])]
        # batch_go = self.embedding_fn(
        #     tf.constant(constant.SYMBOL_GO, dtype=tf.int32, shape=self.model_config.batch_size),
        #     self.data.vocab_simple)
        target_length = len(decode_input_list) + 1
        decoder_emb_inputs = tf.stack(batch_go + decode_input_list, axis=1)
        decoder_output = self.decode_inputs_to_outputs(
            decoder_emb_inputs, encoder_outputs, encoder_attn_bias)
        decoder_output_list = [
            tf.squeeze(d, 1)
            for d in tf.split(decoder_output, target_length, axis=1)]
        return decoder_output_list

    def transformer_beam_search(self, encoder_outputs, encoder_attn_bias):
        # Use Beam Search in evaluation stage
        # Update [a, b, c] to [a, a, a, b, b, b, c, c, c] if beam_search_size == 3
        encoder_beam_outputs = tf.concat(
            [tf.tile(tf.expand_dims(encoder_outputs[o, :, :], axis=0),
                     [self.model_config.beam_search_size, 1, 1])
             for o in range(self.model_config.batch_size)], axis=0)

        encoder_attn_beam_bias = tf.concat(
            [tf.tile(tf.expand_dims(encoder_attn_bias[o, :, :, :], axis=0),
                     [self.model_config.beam_search_size, 1, 1, 1])
             for o in range(self.model_config.batch_size)], axis=0)

        def symbol_to_logits_fn(ids):
            embs = tf.nn.embedding_lookup(self.emb_simple, ids[:, 1:])
            embs = tf.pad(embs, [[0, 0], [1, 0], [0, 0]])
            decoder_outputs = self.decode_inputs_to_outputs(embs, encoder_beam_outputs, encoder_attn_beam_bias)
            return self.output_to_logit(decoder_outputs[:, -1, :])

        beam_ids, beam_score = beam_search.beam_search(symbol_to_logits_fn,
                                                       tf.zeros([self.model_config.batch_size], tf.int32),
                                                       self.model_config.beam_search_size,
                                                       self.model_config.max_simple_sentence,
                                                       len(self.data.vocab_simple.i2w),
                                                       self.model_config.penalty_alpha,
                                                       self.data.vocab_simple.encode(constant.SYMBOL_END))
        top_beam_ids = beam_ids[:, 0, 1:]
        top_beam_ids = tf.pad(top_beam_ids,
                              [[0, 0],
                               [0, self.model_config.max_simple_sentence - tf.shape(top_beam_ids)[1]]])
        decoder_target_list = [tf.squeeze(d, 1)
                               for d in tf.split(top_beam_ids, self.model_config.max_simple_sentence, axis=1)]
        decoder_score = -beam_score[:, 0] / tf.to_float(tf.shape(top_beam_ids)[1])

        # Get outputs based on target ids
        decode_input_embs = tf.stack(self.embedding_fn(decoder_target_list, self.emb_simple), axis=1)
        tf.get_variable_scope().reuse_variables()
        decoder_outputs = self.decode_inputs_to_outputs(decode_input_embs, encoder_outputs, encoder_attn_bias)

        output = ModelOutput(
            decoder_outputs=decoder_outputs,
            decoder_score=decoder_score,
            decoder_target_list=decoder_target_list)
        return output

    def output_to_logit(self, prev_out):
        prev_logit = tf.add(tf.matmul(prev_out, tf.transpose(self.w)), self.b)
        return prev_logit

    def decode_inputs_to_outputs(self, decoder_embed_inputs, encoder_outputs, encoder_attn_bias):
        if self.model_config.decode_input_gate:
            print('Use Decode Input Gate!')
            with tf.variable_scope('decode_input_gate'):
                gate_filter = tf.get_variable('gate_decode',
                                           [1, self.model_config.dimension, self.model_config.dimension],
                                           tf.float32,
                                           initializer=tf.contrib.layers.xavier_initializer())
                gate_decode_input = tf.tanh(
                    tf.nn.conv1d(decoder_embed_inputs, gate_filter, 1, 'SAME'))
                decoder_embed_inputs *= gate_decode_input

        if self.hparams.pos == 'timing':
            decoder_embed_inputs = common_attention.add_timing_signal_1d(decoder_embed_inputs)
            print('Use positional encoding in decoder text.')

        decoder_attn_bias = common_attention.attention_bias_lower_triangle(tf.shape(decoder_embed_inputs)[1])
        decoder_embed_inputs = tf.nn.dropout(decoder_embed_inputs,
                                             1.0 - self.hparams.layer_prepostprocess_dropout)
        if self.model_config.decode_atten_gate:
            print('Use Decode Attention Gate')
            decoder_output = transformer.transformer_decoder_attngate(decoder_embed_inputs,
                                                             encoder_outputs,
                                                             decoder_attn_bias,
                                                             encoder_attn_bias,
                                                             self.hparams)
        elif self.model_config.trans_layer_gate:
            print('Use Layer Gate Transformer')
            decoder_output = transformer.transformer_decoder_gate(decoder_embed_inputs,
                                                             encoder_outputs,
                                                             decoder_attn_bias,
                                                             encoder_attn_bias,
                                                             self.hparams)
        else:
            decoder_output = transformer.transformer_decoder(decoder_embed_inputs,
                                                             encoder_outputs,
                                                             decoder_attn_bias,
                                                             encoder_attn_bias,
                                                             self.hparams)
        return decoder_output

    def setup_hparams(self):
        self.hparams.num_heads = self.model_config.num_heads
        self.hparams.num_hidden_layers = self.model_config.num_hidden_layers
        self.hparams.num_encoder_layers = self.model_config.num_encoder_layers
        self.hparams.num_decoder_layers = self.model_config.num_decoder_layers
        self.hparams.pos = self.model_config.hparams_pos
        self.hparams.hidden_size = self.model_config.dimension
        self.hparams.layer_prepostprocess_dropout = self.model_config.layer_prepostprocess_dropout

        if self.is_train:
            self.hparams.add_hparam('mode', tf.estimator.ModeKeys.TRAIN)
        else:
            self.hparams.add_hparam('mode', tf.estimator.ModeKeys.EVAL)
            self.hparams.layer_prepostprocess_dropout = 0
