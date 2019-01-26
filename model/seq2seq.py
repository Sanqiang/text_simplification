from model.graph import Graph
from model.graph import ModelOutput
from model.seq2seq_beamsearch import beam_search

import tensorflow as tf
from util import constant, nn

class Seq2SeqGraph(Graph):
    def __init__(self, data, is_train, model_config):
        super(Seq2SeqGraph, self).__init__(data, is_train, model_config)
        self.model_fn = self.seq2seq_fn
        self.rand_unif_init = tf.contrib.layers.xavier_initializer() # tf.random_uniform_initializer(-0.05, 0.05, seed=123)
        self.trunc_norm_init = tf.contrib.layers.xavier_initializer() # tf.truncated_normal_initializer(stddev=1e-4)

    def decode_inputs_to_outputs(self, inp, prev_state_c, prev_state_h, encoder_outputs, encoder_padding_bias,
                                 encoder_features, attn_v,
                                 rule_id_input_placeholder, mem_contexts, mem_outputs, global_step):

        def attention(query):
            if self.model_config.attention_type == 'ffn':
                decoder_feature = tf.expand_dims(
                    nn.linear(query, self.model_config.dimension, True, scope='decoder_feature'),
                    axis=1)
                energy = tf.expand_dims(tf.reduce_sum(
                    attn_v * tf.tanh(encoder_features + decoder_feature), axis=2), axis=1)
                energy += encoder_padding_bias
                energy_norm = tf.nn.softmax(energy, axis=2)
                context_vector = tf.matmul(energy_norm, encoder_outputs)
                return tf.squeeze(context_vector, axis=1), tf.squeeze(energy_norm, axis=1)
            elif self.model_config.attention_type == 'dot':
                query = tf.expand_dims(tf.concat(query, axis=1), axis=1)
                weight = tf.matmul(query, encoder_outputs, transpose_b=True)
                weight += encoder_padding_bias
                weight = tf.nn.softmax(weight, axis=2)
                context_vector = tf.matmul(weight, encoder_outputs)
                return tf.squeeze(context_vector, axis=1), tf.squeeze(weight, axis=1)
            elif self.model_config.attention_type == 'bilinear':
                query = tf.expand_dims(tf.concat(query, axis=1), axis=1)
                weight = tf.matmul(query, encoder_features, transpose_b=True)
                weight += encoder_padding_bias
                weight = tf.nn.softmax(weight, axis=2)
                context_vector = tf.matmul(weight, encoder_outputs)
                return tf.squeeze(context_vector, axis=1), tf.squeeze(weight, axis=1)

        prev_state = tf.contrib.rnn.LSTMStateTuple(prev_state_c, prev_state_h)
        if self.is_train:
            inp = tf.nn.dropout(inp,
                          1.0 - self.model_config.layer_prepostprocess_dropout)
        cell_output, state = self.decode_cell(inp, prev_state)
        context_vector, attn_dist = attention(state)
        final_output = nn.linear([context_vector] + [cell_output], self.model_config.dimension, True,
                                 scope='projection')

        cur_context = None
        if 'rule' in self.model_config.memory:
            cur_context = context_vector
            cur_mem_contexts = tf.stack(self.embedding_fn(rule_id_input_placeholder, mem_contexts), axis=1)
            cur_mem_outputs = tf.stack(self.embedding_fn(rule_id_input_placeholder, mem_outputs), axis=1)

            bias = tf.expand_dims(
                -1e9 * tf.to_float(tf.equal(tf.stack(rule_id_input_placeholder, axis=1), 0)),
                axis=1)
            weights = tf.nn.softmax(bias + tf.matmul(tf.expand_dims(context_vector, axis=1), cur_mem_contexts, transpose_b=True))
            mem_output = tf.squeeze(tf.matmul(weights, cur_mem_outputs), axis=1)
            nmem_output = nn.linear([final_output] + [mem_output], self.model_config.dimension, True)
            g = tf.greater(global_step, tf.constant(2 * self.model_config.memory_prepare_step, dtype=tf.int64))
            final_output = tf.cond(g, lambda: nmem_output, lambda: final_output)

        return final_output, state[0], state[1], attn_dist, cur_context


    def seq2seq_fn(self, sentence_complex_input_placeholder, emb_complex,
                   sentence_simple_input_placeholder, emb_simple,
                   w, b, rule_id_input_placeholder, mem_contexts, mem_outputs, global_step):
        train_mode = self.model_config.train_mode
        with tf.variable_scope('seq2seq_encoder'):
            encoder_embed_inputs = tf.stack(
                self.embedding_fn(sentence_complex_input_placeholder, emb_complex), axis=1)
            encoder_len = tf.cast(tf.reduce_sum(tf.to_float(tf.not_equal(tf.stack(sentence_complex_input_placeholder, axis=1),
                                 self.data.vocab_complex.encode(constant.SYMBOL_PAD))), axis=1), tf.int32)

            cell_fw = tf.contrib.rnn.LSTMCell(self.model_config.dimension, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self.model_config.dimension, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_embed_inputs,
                                                                                dtype=tf.float32,
                                                                                sequence_length=encoder_len,
                                                                                swap_memory=False)
            encoder_outputs = tf.concat(axis=2, values=encoder_outputs)
            encoder_padding_bias = tf.expand_dims(tf.to_float(tf.equal(tf.stack(sentence_complex_input_placeholder, axis=1),
                                 self.data.vocab_complex.encode(constant.SYMBOL_PAD))), axis=1) * -1e9

            # variables prepare for decoder
            encoder_features, attn_v = None, None
            if self.model_config.attention_type == 'bilinear':
                attn_w = tf.get_variable("attn_w",
                                         [1, 2 * self.model_config.dimension, 2 * self.model_config.dimension])
                encoder_features = tf.nn.conv1d(encoder_outputs, attn_w, 1, 'SAME')
            elif self.model_config.attention_type == 'ffn':
                attn_w = tf.get_variable("attn_w",
                                         [1, 2 * self.model_config.dimension, self.model_config.dimension])
                encoder_features = tf.nn.conv1d(encoder_outputs, attn_w, 1, 'SAME')
                attn_v = tf.get_variable("v", [1, 1, self.model_config.dimension])

            w_reduce_c = tf.get_variable('w_reduce_c',
                                         [self.model_config.dimension * 2, self.model_config.dimension],
                                         dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable('w_reduce_h',
                                         [self.model_config.dimension * 2, self.model_config.dimension],
                                         dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c', [self.model_config.dimension], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable('bias_reduce_h', [self.model_config.dimension], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)

            # Apply linear layer
            old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])  # Concatenation of fw and bw cell
            old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])  # Concatenation of fw and bw state
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)  # Get new cell from old cell
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)  # Get new state from old state
            decoder_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        with tf.variable_scope('seq2seq_decoder'):
            self.decode_cell = tf.contrib.rnn.LSTMCell(
                self.model_config.dimension, state_is_tuple=True, initializer=self.rand_unif_init)
            state = decoder_in_state

            if self.is_train:
                batch_go = tf.zeros(
                    [self.model_config.batch_size, self.model_config.dimension])
                decoder_embed_inputs_list = self.embedding_fn(
                    sentence_simple_input_placeholder[:-1], emb_simple)
                decoder_embed_inputs_list = [batch_go] + decoder_embed_inputs_list

                outputs = []
                logits = []
                attn_dists = []
                targats = []
                contexts = []
                sampled_targets = []
                sampled_logits = []
                for i, dec_inp in enumerate(decoder_embed_inputs_list):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    x = dec_inp
                    state_c, state_h = state[0], state[1]
                    final_output, state_c, state_h, attn_dist, context = self.decode_inputs_to_outputs(
                        x, state_c, state_h, encoder_outputs, encoder_padding_bias,
                        encoder_features, attn_v,
                        rule_id_input_placeholder, mem_contexts, mem_outputs, global_step)
                    state = tf.contrib.rnn.LSTMStateTuple(state_c, state_h)

                    logit = self.output_to_logit(final_output, w, b)
                    target = tf.argmax(logit, axis=1)

                    if train_mode == 'dynamic_self-critical':
                        target = tf.stop_gradient(target)
                        sampled_target = tf.cast(tf.squeeze(
                            tf.multinomial(logit, 1), axis=1), tf.int32)
                        indices = tf.stack(
                            [tf.range(0, self.model_config.batch_size, dtype=tf.int32),
                             tf.squeeze(sampled_target)],
                            axis=-1)
                        sampled_logit = tf.gather_nd(tf.nn.softmax(logit, axis=1), indices)
                        sampled_targets.append(sampled_target)
                        sampled_logits.append(sampled_logit)

                    targats.append(target)
                    logits.append(logit)
                    outputs.append(final_output)
                    attn_dists.append(attn_dist)
                    contexts.append(context)

                if 'rule' in self.model_config.memory:
                    contexts = tf.stack(contexts, axis=1)
                gt_target_list = sentence_simple_input_placeholder
                output = ModelOutput(
                    contexts=contexts if 'rule' in self.model_config.memory else None,
                    encoder_outputs=encoder_outputs,
                    decoder_outputs_list=outputs,
                    final_outputs_list=outputs,
                    decoder_logit_list=logits,
                    gt_target_list=gt_target_list,
                    encoder_embed_inputs_list=tf.unstack(encoder_embed_inputs, axis=1),
                    decoder_target_list=targats,
                    sample_logit_list=sampled_logits if train_mode == 'dynamic_self-critical' else None,
                    sample_target_list=sampled_targets if train_mode == 'dynamic_self-critical' else None,
                    decoder_score=0.0,
                    attn_distr_list=attn_dists,
                )
                self.attn_dists = tf.stack(attn_dists, axis=1)
                self.targets = tf.stack(targats, axis=1)
                self.cs = tf.stack(sentence_complex_input_placeholder, axis=1)
                return output
            else:
                # encoder_beam_outputs = tf.concat(
                #     [tf.tile(tf.expand_dims(encoder_outputs[o, :, :], axis=0),
                #              [self.model_config.beam_search_size, 1, 1])
                #      for o in range(self.model_config.batch_size)], axis=0)
                #
                # encoder_beam_features = tf.concat(
                #     [tf.tile(tf.expand_dims(encoder_outputs[o, :, :], axis=0),
                #              [self.model_config.beam_search_size, 1, 1])
                #      for o in range(self.model_config.batch_size)], axis=0)
                #
                # encoder_beam_padding_bias = tf.concat(
                #     [tf.tile(tf.expand_dims(encoder_padding_bias[o, :], axis=0),
                #              [self.model_config.beam_search_size, 1])
                #      for o in range(self.model_config.batch_size)], axis=0)

                def symbol_to_logits_fn(ids, pre_state_c, pre_state_h):
                    id = ids[:, -1]
                    inp = self.embedding_fn(id, emb_simple)
                    final_output, state_c, state_h, attn_dist, _ = self.decode_inputs_to_outputs(
                        inp, pre_state_c, pre_state_h, encoder_outputs, encoder_padding_bias, encoder_features, attn_v,
                        rule_id_input_placeholder, mem_contexts, mem_outputs, global_step)
                    logit = self.output_to_logit(final_output, w, b)
                    return logit, state_c, state_h, attn_dist

                beam_ids, beam_score, beam_attn_distrs = beam_search(symbol_to_logits_fn,
                                                   tf.zeros([self.model_config.batch_size], tf.int32),
                                                   self.model_config.beam_search_size,
                                                   self.model_config.max_simple_sentence,
                                                   self.data.vocab_simple.vocab_size(),
                                                   self.model_config.penalty_alpha,
                                                   state[0], state[1],
                                                   model_config=self.model_config)
                top_beam_ids = beam_ids[:, 0, 1:]
                top_beam_ids = tf.pad(top_beam_ids,
                                      [[0, 0],
                                       [0, self.model_config.max_simple_sentence - tf.shape(top_beam_ids)[1]]])
                decoder_target_list = [tf.squeeze(d, 1)
                                       for d in tf.split(top_beam_ids, self.model_config.max_simple_sentence, axis=1)]
                decoder_score = -beam_score[:, 0] / tf.to_float(tf.shape(top_beam_ids)[1])

                top_attn_distrs = beam_attn_distrs[:, 0, 1:]
                top_attn_distrs = tf.pad(top_attn_distrs,
                                      [[0, 0],
                                       [0, self.model_config.max_simple_sentence - tf.shape(top_attn_distrs)[1]], [0,0]])
                top_attn_distrs.set_shape(
                    [self.model_config.batch_size, self.model_config.max_simple_sentence, self.model_config.max_complex_sentence])
                gt_target_list = sentence_simple_input_placeholder
                output = ModelOutput(
                    # contexts=cur_context if 'rule' in self.model_config.memory else None,
                    encoder_outputs=encoder_outputs,
                    # decoder_outputs_list=outputs if train_mode != 'dynamic_self-critical' else None,
                    # final_outputs_list=outputs if train_mode != 'dynamic_self-critical' else None,
                    # decoder_logit_list=logits if train_mode != 'dynamic_self-critical' else None,
                    gt_target_list=gt_target_list,
                    # encoder_embed_inputs_list=tf.unstack(encoder_embed_inputs, axis=1),
                    decoder_target_list=decoder_target_list,
                    # sample_logit_list=sampled_logit_list if train_mode == 'dynamic_self-critical' else None,
                    # sample_target_list=sampled_target_list if train_mode == 'dynamic_self-critical' else None,
                    decoder_score=decoder_score,
                    attn_distr_list=top_attn_distrs
                )
                return output



