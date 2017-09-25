"""Seq2Seq Computational Graph
   based on https://github.com/abisee/pointer-generator/blob/master/model.py"""
import tensorflow as tf
import numpy as np

from util import constant
from model.graph import Graph
from util.nn import linear

class Seq2SeqGraph(Graph):
    def __init__(self, data, is_train, model_config):
        super(Seq2SeqGraph, self).__init__(data, is_train, model_config)
        self.model_fn = self.seq2seq_fn


    def seq2seq_fn(self):
        self._enc_padding_mask = tf.stack(
            [tf.to_float(tf.not_equal(d, self.data.vocab_simple.encode(constant.SYMBOL_PAD)))
             for d in self.sentence_complex_input_placeholder], axis=1)
        self._enc = self.embedding_fn(self.sentence_complex_input_placeholder, self.emb_complex)
        self._enc_len = tf.to_int32(tf.reduce_mean(self._enc_padding_mask, axis=1))

        enc_outputs, fw_st, bw_st = self._encoder()
        self._enc_states = enc_outputs

        self._dec_in_state = self._reduce_states(fw_st, bw_st)

        with tf.variable_scope('decoder'):
            decoder_outputs, self._dec_out_state, self.attn_dists = self._decoder()

        logits = []
        for i, output in enumerate(decoder_outputs):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            logits.append(tf.nn.xw_plus_b(output, tf.transpose(self.w), self.b))
        logits = [tf.nn.softmax(s) for s in logits]

        if not self.is_train:
            assert len(logits) == 1
            final_dists = logits[0]
            topk_probs, self._topk_ids = tf.nn.top_k(
                final_dists, self.model_config.beam_search_size)
            self._topk_log_probs = tf.log(topk_probs)

        return decoder_outputs, logits, self.sentence_simple_input_placeholder

    def _decoder(self):
        att_size = self._enc_states.get_shape()[-1].value
        cell = tf.contrib.rnn.LSTMCell(
            att_size, state_is_tuple=True, initializer=self.rand_unif_init)
        with tf.variable_scope('attention_decoder') as scope:
            encoder_states = tf.expand_dims(self._enc_states, axis=2)


            # Attention-based feature vector
            W_h = tf.get_variable("W_h",
                                  [1, 1, att_size, att_size])
            encoder_features = tf.nn.conv2d(
                encoder_states, W_h, [1, 1, 1, 1], 'SAME')
            v = tf.get_variable("v", [att_size])

            def attention(decoder_state):
                with tf.variable_scope("attention"):
                    def masked_attention(e):
                        attn_dist = tf.nn.softmax(e)  # take softmax. shape (batch_size, attn_length)
                        attn_dist *= self._enc_padding_mask  # apply mask
                        masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                        return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize

                    decoder_features = linear(decoder_state, att_size, True)
                    decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)
                    e = tf.reduce_sum(v * tf.tanh(encoder_features + decoder_features), [2, 3])
                    attn_dist = masked_attention(e)
                    context_vector = tf.reduce_sum(
                        tf.reshape(attn_dist, [self.model_config.batch_size, -1, 1, 1]) * encoder_states,
                        [1, 2])  # shape (batch_size, attn_size).
                    context_vector = tf.reshape(context_vector, [-1, att_size])

                return context_vector, attn_dist

            outputs = []
            attn_dists = []
            state = self._dec_in_state
            context_vector = tf.zeros((self.model_config.batch_size, att_size))
            if not self.is_train:
                context_vector, _ = attention(self._dec_in_state)
            for i, inp in enumerate(
                    self.embedding_fn(self.sentence_simple_input_placeholder, self.emb_simple)):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                x = linear([inp] + [context_vector], att_size, True)
                cell_output, state = cell(x, state)
                if i == 0 and not self.is_train:
                    with tf.variable_scope(
                            tf.get_variable_scope(), reuse=True):
                        context_vector, attn_dist = attention(state)
                else:
                    context_vector, attn_dist = attention(state)
                attn_dists.append(attn_dist)

                with tf.variable_scope("AttnOutputProjection"):
                    output = linear([cell_output] + [context_vector], cell.output_size, True)
                outputs.append(output)
            return outputs, state, attn_dists

    def _encoder(self):
        with tf.variable_scope('encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(
                self.model_config.dimension,
                initializer=self.rand_unif_init, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(
                self.model_config.dimension,
                initializer=self.rand_unif_init, state_is_tuple=True)
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, tf.stack(self._enc, axis=1),
                dtype=tf.float32,
                sequence_length=self._enc_len,
                swap_memory=True)
            encoder_outputs = tf.concat(axis=2, values=encoder_outputs)
            return encoder_outputs, fw_st, bw_st

    def _reduce_states(self, fw_st, bw_st):
        with tf.variable_scope('reduce_final_st'):
            # Define weights and biases to reduce the cell and reduce the state
            w_reduce_c = tf.get_variable(
                'w_reduce_c', [self.model_config.dimension * 2, self.model_config.dimension * 2],
                dtype=tf.float32, initializer=self.rand_unif_init)
            w_reduce_h = tf.get_variable(
                'w_reduce_h', [self.model_config.dimension * 2, self.model_config.dimension * 2],
                dtype=tf.float32, initializer=self.rand_unif_init)
            bias_reduce_c = tf.get_variable(
                'bias_reduce_c', [self.model_config.dimension * 2],
                dtype=tf.float32, initializer=self.rand_unif_init)
            bias_reduce_h = tf.get_variable(
                'bias_reduce_h', [self.model_config.dimension * 2],
                dtype=tf.float32, initializer=self.rand_unif_init)

            # Apply linear layer
            old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])
            old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)
            return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

    # For Evaluation Beam Search Graph
    # Following code will only dedicated to beam search in evaluation.

    def run_encoder(self, sess, input_feed):
        fetches = [self._enc_states, self._dec_in_state, self.global_step]
        (enc_states, dec_in_state, global_step) = sess.run(
            fetches, input_feed)
        return enc_states, dec_in_state

    def beam_search(self, sess, input_feed):
        enc_states, dec_in_state = self.run_encoder(sess, input_feed)
        hyps = [BeamSearch_Hypothesis(
            tokens=[self.data.vocab_simple.encode(constant.SYMBOL_START)],
            log_probs=[0.0],
            state=dec_in_state,
            attn_dists=[]) for _ in range(self.model_config.beam_search_size)]

        results = []
        steps = 0
        while steps < self.model_config.max_simple_sentence and len(results) < self.model_config.beam_search_size:
            latest_tokens = [h.latest_token for h in hyps]
            states = [h.state for h in hyps]
            (topk_ids, topk_log_probs, new_states, attn_dists) = self.beam_search_onestep(
                sess, latest_tokens, enc_states, states, input_feed)

            all_hyps = []
            num_orig_hyps = 1 if steps == 0 else len(hyps)
            for i in range(num_orig_hyps):
                h, new_state, attn_dist = hyps[i], new_states[i], attn_dists[i]
                for j in range(self.model_config.beam_search_size * 2):
                    new_hyp = h.extend(token=topk_ids[i, j],
                                       log_prob=topk_log_probs[i, j],
                                       state=new_state,
                                       attn_dist=attn_dist)
                    all_hyps.append(new_hyp)

            hyps = []
            for h in self.sort_hyps(all_hyps):
                if h.latest_token == self.data.vocab_simple.encode(constant.SYMBOL_END):
                    if steps >= self.model_config.min_simple_sentence:
                        results.append(h)
                else:
                    hyps.append(h)
                if len(hyps) == self.model_config.beam_search_size or len(results) == self.model_config.beam_search_size:
                    break
                steps += 1

        if len(results) == 0:
            results = hyps

        hyps_sorted = self.sort_hyps(results)
        return hyps_sorted[0]

    def sort_hyps(self, hyps):
        return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)

    def beam_search_onestep(self, sess, latest_tokens, enc_states, dec_init_states, input_feed):
        cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
        hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
        new_c = np.concatenate(cells, axis=0)
        new_h = np.concatenate(hiddens, axis=0)
        new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        feed = {
            self._enc_states: enc_states,
            self._dec_in_state: new_dec_in_state,
            self.sentence_simple_input_placeholder: np.transpose(np.array([latest_tokens])),
        }

        to_return = {
            "ids": self._topk_ids,
            "probs": self._topk_log_probs,
            "states": self._dec_out_state,
            "attn_dists": self.attn_dists
        }

        results = sess.run(to_return, feed_dict=feed)
        new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in
                      range(self.model_config.beam_search_size)]
        assert len(results['attn_dists']) == 1
        attn_dists = results['attn_dists'][0].tolist()

        return results['ids'], results['probs'], new_states, attn_dists

class BeamSearch_Hypothesis:
    def __init__(self, tokens, log_probs, state, attn_dists):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.attn_dists = attn_dists

    def extend(self, token, log_prob, state, attn_dist):
        return BeamSearch_Hypothesis(tokens=self.tokens + [token],
                                     log_probs=self.log_probs + [log_prob],
                                     state=state,
                                     attn_dists=self.attn_dists + [attn_dist])

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        return self.log_prob / len(self.tokens)
