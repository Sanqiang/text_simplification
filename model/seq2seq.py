"""Seq2Seq Computational Graph
   based on https://github.com/abisee/pointer-generator/blob/master/model.py"""
import tensorflow as tf

from model.graph import Graph
from model.graph import ModelOutput
from util import constant
from util.nn import linear


class Seq2SeqGraph(Graph):
    def __init__(self, data, is_train, model_config):
        super(Seq2SeqGraph, self).__init__(data, is_train, model_config)
        self.model_fn = self.seq2seq_fn

    def seq2seq_fn(self):
        # Add <go> in front, and truncate last one.
        self.sentence_simple_input = ([tf.zeros(
            self.model_config.batch_size, tf.int32, name='simple_input_go')] +
                                 self.sentence_simple_input_placeholder[:-1])
        self.sentence_simple_output = self.sentence_simple_input_placeholder

        self._enc_padding_mask = tf.stack(
            [tf.to_float(tf.not_equal(d, self.data.vocab_simple.encode(constant.SYMBOL_PAD)))
             for d in self.sentence_complex_input_placeholder], axis=1)
        self._enc = self.embedding_fn(self.sentence_complex_input_placeholder, self.emb_complex)
        self._enc_len = tf.to_int32(tf.reduce_sum(self._enc_padding_mask, axis=1))

        enc_outputs, fw_st, bw_st = self._encoder()
        self._enc_states = enc_outputs

        self._dec_in_state = self._reduce_states(fw_st, bw_st)

        with tf.variable_scope('decoder'):
            decoder_outputs, logits, self._dec_out_state, self.attn_dists = self._decoder()

        if not self.is_train and self.model_config.beam_search_size > 0:
            # assert len(logits) == 1
            self.final_dists = logits[-1]
            topk_probs, self._topk_ids = tf.nn.top_k(
                self.final_dists, self.model_config.beam_search_size * 2)
            self._topk_log_probs = tf.log(topk_probs)

        self.logits = logits

        output = ModelOutput(
            decoder_outputs=decoder_outputs,
            decoder_logit_list=logits,
            gt_target_list=self.sentence_simple_output
        )
        return output

    def _decoder(self):
        hidden_size = self._enc_states.get_shape()[-1].value
        cell = tf.contrib.rnn.LSTMCell(
            self.model_config.dimension, state_is_tuple=True, initializer=self.rand_unif_init)
        with tf.variable_scope('attention_decoder'):
            encoder_states = tf.expand_dims(self._enc_states, axis=2)
            # encoder_memories = tf.expand_dims(tf.stack(
            #     self.embedding_fn(self.sentence_complex_input_placeholder, self.emb_complex), axis=1), axis=2)

            # Attention-based feature vector
            W_h = tf.get_variable("W_h",
                                  [1, 1, hidden_size, self.model_config.dimension])
            encoder_features = tf.nn.conv2d(
                encoder_states, W_h, [1, 1, 1, 1], 'SAME')
            v = tf.get_variable("v", [self.model_config.dimension])

            def attention(query):
                with tf.variable_scope("attention"):
                    decoder_features = linear(query, self.model_config.dimension, True)
                    decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)

                    def masked_attention(e):
                        attn_dist = tf.nn.softmax(e)
                        # attn_dist *= self._enc_padding_mask
                        masked_sums = tf.reduce_sum(attn_dist, axis=1)
                        return attn_dist / tf.reshape(masked_sums, [-1, 1])

                    e = tf.reduce_sum(v * tf.tanh(encoder_features + decoder_features), [2, 3])
                    attn_dist = masked_attention(e)
                    context_vector = tf.reduce_sum(
                        tf.reshape(attn_dist, [self.model_config.batch_size, -1, 1, 1]) * encoder_features,
                        [1, 2])  # shape (batch_size, attn_size).
                return context_vector, attn_dist

            outputs = []
            logits = []
            attn_dists = []
            state = self._dec_in_state
            context_vector, _ = attention(state)
            gt_inp_embs = self.embedding_fn(self.sentence_simple_input, self.emb_simple)
            for i in range(self.model_config.max_simple_sentence):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                    if not self.is_train:
                        inp_idx = tf.cast(tf.argmax(logit, axis=1), tf.int32)
                        inp_emb = self.embedding_fn([inp_idx], self.emb_simple)[0]
                if self.is_train or i == 0:
                    inp_emb = gt_inp_embs[i]

                x = linear([inp_emb] + [context_vector], self.model_config.dimension, True)
                cell_output, state = cell(x, state)
                if i == 0:
                    with tf.variable_scope(
                            tf.get_variable_scope(), reuse=True):
                        context_vector, attn_dist = attention(state)
                else:
                    context_vector, attn_dist = attention(state)
                attn_dists.append(attn_dist)

                with tf.variable_scope("AttnOutputProjection"):
                    output = linear([cell_output] + [context_vector], cell.output_size, True)
                    logit = tf.nn.xw_plus_b(output, tf.transpose(self.w), self.b)
                outputs.append(output)
                logits.append(logit)
            return outputs, logits, state, attn_dists

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
                'w_reduce_c', [self.model_config.dimension * 2, self.model_config.dimension],
                dtype=tf.float32, initializer=self.rand_unif_init)
            w_reduce_h = tf.get_variable(
                'w_reduce_h', [self.model_config.dimension * 2, self.model_config.dimension],
                dtype=tf.float32, initializer=self.rand_unif_init)
            bias_reduce_c = tf.get_variable(
                'bias_reduce_c', [self.model_config.dimension],
                dtype=tf.float32, initializer=self.rand_unif_init)
            bias_reduce_h = tf.get_variable(
                'bias_reduce_h', [self.model_config.dimension],
                dtype=tf.float32, initializer=self.rand_unif_init)

            # Apply linear layer
            old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])
            old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)
            return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

    # For Evaluation Beam Search Graph
    # Following code will only dedicated to beam search in evaluation.

    def run_encoder(self, sess, graph, input_feed):
        fetches = [graph._enc_states, graph._dec_in_state, graph.global_step]
        (enc_states, dec_in_state, global_step) = sess.run(
            fetches, input_feed)
        return enc_states, dec_in_state, global_step

    def beam_search(self, sess, graph, input_feed):
        enc_states, dec_in_state, global_step = self.run_encoder(sess, graph, input_feed)
        hyps = [BeamSearch_Hypothesis(
            tokens=[self.data.vocab_simple.encode(constant.SYMBOL_START)],
            log_probs=[0.0],
            state=dec_in_state,
            attn_dists=[]) for _ in range(self.model_config.beam_search_size)]

        results = []
        steps = 0
        while steps < self.model_config.max_simple_sentence and len(results) < self.model_config.beam_search_size:
            topk_ids = []
            topk_log_probs = []
            new_states = []
            attn_dists = []
            for h in hyps:
                (topk_id, topk_log_prob, new_state, attn_dist) = self.beam_search_onestep(
                    sess, h.latest_token, enc_states, h.state, graph, input_feed)
                topk_ids.append(topk_id[0])
                topk_log_probs.append(topk_log_prob[0])
                new_states.append(new_state)
                attn_dists.append(attn_dist)

            all_hyps = []
            num_orig_hyps = 1 if steps == 0 else len(hyps)
            for i in range(num_orig_hyps):
                h, new_state, attn_dist = hyps[i], new_states[i], attn_dists[i]
                for j in range(self.model_config.beam_search_size * 2):
                    new_hyp = h.extend(token=topk_ids[i][j],
                                       log_prob=topk_log_probs[i][j],
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
        return hyps_sorted[0], global_step

    def sort_hyps(self, hyps):
        return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)

    def beam_search_onestep(self, sess, latest_token, enc_state, dec_init_state, graph, input_feed):
        feed = {
            graph._enc_states: enc_state,
            graph._dec_in_state: dec_init_state,
        }

        feed[graph.sentence_simple_input[0].name] = [latest_token]

        to_return = {
            "ids": graph._topk_ids,
            "probs": graph._topk_log_probs,
            "states": graph._dec_out_state,
            "attn_dists": graph.attn_dists,
            'final_dists':graph.final_dists
        }

        results = sess.run(to_return, feed_dict=feed)
        new_states = tf.contrib.rnn.LSTMStateTuple(results['states'].c, results['states'].h)
        # assert len(results['attn_dists']) == 1
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
