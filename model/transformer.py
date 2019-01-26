import tensorflow as tf
from tensor2tensor.layers import common_attention, common_layers
from tensor2tensor.models import transformer
from tensor2tensor.models.research import universal_transformer_util, universal_transformer
from tensor2tensor.utils import beam_search

from util import constant
from model.graph import Graph
from model.graph import ModelOutput
from model.copy_mechanism import word_distribution
from model.bert.modeling import BertModel, BertConfig


class TransformerGraph(Graph):
    def __init__(self, data, is_train, model_config):
        super(TransformerGraph, self).__init__(data, is_train, model_config)
        self.hparams = transformer.transformer_base()
        self.setup_hparams()
        self.model_fn = self.transformer_fn

    def update_score(self, score, encoder_outputs=None, encoder_mask=None, comp_features=None):
        if 'pred' not in self.model_config.tune_mode and 'cond' not in self.model_config.tune_mode:
            return score, None

        if 'pred' in self.model_config.tune_mode:
            # TODO(sanqiang): change pred mode into better prediction
            # In pred mode, the scores are only factor to multiply
            dimension_unit = int(self.model_config.dimension / 3)
            dimension_runit = self.model_config.dimension - 2 * dimension_unit
            ppdb_multiplier = tf.expand_dims(score[:, :, 0], axis=-1)
            add_multipler = tf.expand_dims(score[:, :, dimension_unit], axis=-1)
            len_multipler = tf.expand_dims(score[:, :, dimension_unit*2], axis=-1)

            evidence = tf.stop_gradient(encoder_outputs)
            evidence_mask = tf.stop_gradient(encoder_mask)
            evidence = tf.reduce_sum(evidence*tf.expand_dims(evidence_mask, axis=-1), axis=1)\
                       / (1.0 + tf.expand_dims(tf.reduce_sum(evidence_mask, axis=1), axis=-1))

            ppdb_pred_score = tf.squeeze(tf.contrib.layers.fully_connected(evidence, 1, scope='ppdb_pred_score'), axis=-1)
            add_pred_score = tf.squeeze(tf.contrib.layers.fully_connected(evidence, 1, scope='add_pred_score'), axis=-1)
            len_pred_score = tf.squeeze(tf.contrib.layers.fully_connected(evidence, 1, scope='len_pred_score'), axis=-1)

            if self.is_train:
                # In training, score are from fetch data
                apply_score = score
            else:
                # In evaluating/predict, scores are factor to multiply
                ppdb_score = tf.expand_dims(tf.tile(
                    tf.expand_dims(ppdb_pred_score, axis=-1),
                    [1, dimension_unit]), axis=1) * ppdb_multiplier
                add_score = tf.expand_dims(tf.tile(
                    tf.expand_dims(add_pred_score, axis=-1),
                    [1, dimension_unit]), axis=1) * add_multipler
                len_score = tf.expand_dims(tf.tile(
                    tf.expand_dims(len_pred_score, axis=-1),
                    [1, dimension_runit]), axis=1) * len_multipler
                apply_score = tf.concat([ppdb_score, add_score, len_score], axis=-1)
            # apply_score = tf.Print(apply_score, [ppdb_multiplier, add_multipler, len_multipler, apply_score],
            #                        message='Update multipler for 3 styles:', first_n=-1, summarize=100)
            return apply_score, (ppdb_pred_score, add_pred_score, len_pred_score)

        elif 'cond' in self.model_config.tune_mode:
            print('In eval, the tune scores are based on normal sentence.')
            tune_cnt = 0
            scores = [False, False, False, False]
            if self.model_config.tune_style[0]:
                scores[0] = True
                tune_cnt += 1
            if self.model_config.tune_style[1]:
                scores[1] = True
                tune_cnt += 1
            if self.model_config.tune_style[2]:
                scores[2] = True
                tune_cnt += 1
            if self.model_config.tune_style[3]:
                scores[3] = True
                tune_cnt += 1

            dimension_unit = int(self.model_config.dimension / tune_cnt)
            dimension_runit = self.model_config.dimension - (tune_cnt-1) * dimension_unit
            apply_scores = []
            if self.model_config.tune_style[0]:
                ppdb_score = tf.expand_dims(tf.tile(
                    tf.expand_dims(tf.constant(
                        self.model_config.tune_style[0],
                        shape=[self.model_config.batch_size], dtype=tf.float32), axis=-1),
                    [1, dimension_unit]), axis=1)
                print('Create PPDB score %s' % ppdb_score)
                apply_scores.append(ppdb_score)

            if self.model_config.tune_style[1]:
                dsim_score = tf.expand_dims(tf.tile(
                    tf.expand_dims(tf.constant(
                        self.model_config.tune_style[1],
                        shape=[self.model_config.batch_size], dtype=tf.float32), axis=-1),
                    [1, dimension_unit]), axis=1)
                print('Create Dsim score %s' % dsim_score)
                apply_scores.append(dsim_score)

            if self.model_config.tune_style[2]:
                add_multipler = tf.expand_dims(score[:, :, dimension_unit], axis=-1)
                add_score = tf.expand_dims(tf.tile(
                    tf.expand_dims(comp_features['comp_add_score'], axis=-1),
                    [1, dimension_unit]), axis=1) * add_multipler
                print('Create ADD score %s' % add_score)
                apply_scores.append(add_score)

            if self.model_config.tune_style[3]:
                len_multipler = tf.expand_dims(score[:, :, dimension_unit * 2], axis=-1)
                dimension = dimension_runit if tune_cnt == 3 else dimension_unit
                len_score = tf.expand_dims(tf.tile(
                    tf.expand_dims(comp_features['comp_length'], axis=-1),
                    [1, dimension]), axis=1) * len_multipler
                print('Create LEN score %s' % len_score)
                apply_scores.append(len_score)

            apply_score = tf.concat(apply_scores, axis=-1)
            print('Update apply socre %s' % apply_score)

            return apply_score, None

    def update_encoder_embedding(self, input_embedding, score):
        if not self.model_config.tune_style or 'encoder' not in self.model_config.tune_mode:
            return input_embedding

        embedding_start = tf.slice(input_embedding, [0, 0, 0], [-1, 1, -1])
        embedding_start *= score
        embedding_rest = tf.slice(input_embedding, [0, 1, 0], [-1, -1, -1])
        output_embedding = tf.concat([embedding_start, embedding_rest], axis=1)
        print('Update embedding for encoder.')
        return output_embedding

    def update_decoder_embedding(self, input_embedding, score, beam_size=None):
        if not self.model_config.tune_style or 'decoder' not in self.model_config.tune_mode:
            return input_embedding

        if beam_size and not self.is_train:
            score = tf.tile(score, [1, beam_size, 1])
            score = tf.reshape(score, [-1, 1, self.model_config.dimension])

        embedding_start = tf.slice(input_embedding, [0, 0, 0], [-1, 1, -1])
        embedding_start *= score
        embedding_rest = tf.slice(input_embedding, [0, 1, 0], [-1, -1, -1])
        output_embedding = tf.concat([embedding_start, embedding_rest], axis=1)
        print('Update embedding for decoder.')

        return output_embedding

    def transformer_fn(self,
                       sentence_complex_input_placeholder, emb_complex,
                       sentence_simple_input_placeholder, emb_simple,
                       w, b,
                       rule_id_input_placeholder, rule_target_input_placeholder,
                       mem_contexts, mem_outputs,
                       global_step, score, comp_features, obj):
        encoder_mask = tf.to_float(
            tf.equal(tf.stack(sentence_complex_input_placeholder, axis=1),
                     self.data.vocab_complex.encode(constant.SYMBOL_PAD)))
        encoder_attn_bias = common_attention.attention_bias_ignore_padding(encoder_mask)

        obj_tensors = {}

        train_mode = self.model_config.train_mode
        if self.model_config.bert_mode:
            # Leave space for decoder when static seq
            gpu_id = 0 if train_mode == 'static_seq' or train_mode == 'static_self-critical' or 'direct' in self.model_config.memory else 1
            with tf.device('/device:GPU:%s' % gpu_id):
                sentence_complex_input = tf.stack(sentence_complex_input_placeholder, axis=1)
                bert_model = BertModel(
                    BertConfig.from_json_file(self.model_config.bert_config),
                    self.is_train, sentence_complex_input,
                    input_mask=1.0-encoder_mask, token_type_ids=None, use_one_hot_embeddings=False)
                encoder_embed_inputs = bert_model.embedding_output
                encoder_outputs = bert_model.sequence_output
                emb_complex = bert_model.embedding_table # update emb complex
                if (self.model_config.tie_embedding == 'all' or
                        self.model_config.tie_embedding == 'enc_dec'):
                    emb_simple = bert_model.embedding_table
                if (self.model_config.tie_embedding == 'all' or
                        self.model_config.tie_embedding == 'dec_out'):
                    emb_w_proj = tf.get_variable(
                        'emb_w_proj', shape=[self.model_config.dimension, self.model_config.dimension],
                        initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                    w = tf.matmul(bert_model.embedding_table, emb_w_proj)

                if 'direct' in self.model_config.memory:
                    with tf.device('/device:GPU:1'):
                        direct_mask = tf.to_float(
                            tf.equal(tf.stack(rule_target_input_placeholder, axis=1),
                                     self.data.vocab_complex.encode(constant.SYMBOL_PAD)))
                        direct_bert_model = BertModel(
                            BertConfig.from_json_file(self.model_config.bert_config),
                            self.is_train, tf.stack(rule_target_input_placeholder, axis=1),
                            input_mask=1.0 - direct_mask, token_type_ids=None, use_one_hot_embeddings=False,
                            embedding_table=emb_simple,
                            scope='direct')
                        direct_bert_output = direct_bert_model.sequence_output
                        obj_tensors['direct_bert_bias'] = common_attention.attention_bias_ignore_padding(direct_mask)
                        obj_tensors['direct_bert_output'] = direct_bert_output
        else:
            encoder_embed_inputs = tf.stack(
                self.embedding_fn(sentence_complex_input_placeholder, emb_complex), axis=1)
            if self.hparams.pos == 'timing':
                encoder_embed_inputs = common_attention.add_timing_signal_1d(encoder_embed_inputs)
                print('Use positional encoding in encoder text.')

            if self.model_config.subword_vocab_size and self.model_config.seg_mode:
                encoder_embed_inputs = common_attention.add_positional_embedding(
                    encoder_embed_inputs, 100, 'seg_embedding',
                    positions=obj['line_comp_segids'])
                print('Add segment embedding.')

            with tf.variable_scope('transformer_encoder'):
                encoder_embed_inputs = tf.nn.dropout(encoder_embed_inputs,
                                                     1.0 - self.hparams.layer_prepostprocess_dropout)

                if self.model_config.architecture == 'ut2t':
                    encoder_outputs, encoder_extra_output = universal_transformer_util.universal_transformer_encoder(
                        encoder_embed_inputs, encoder_attn_bias, self.hparams)
                    enc_ponder_times, enc_remainders = encoder_extra_output
                    extra_encoder_loss = (
                            self.hparams.act_loss_weight *
                            tf.reduce_mean(enc_ponder_times + enc_remainders))
                    if self.is_train:
                        obj_tensors['extra_encoder_loss'] = extra_encoder_loss
                else:
                    encoder_outputs = transformer.transformer_encoder(
                        encoder_embed_inputs, encoder_attn_bias, self.hparams)

                # Update score based on multiplier
                score, pred_score_tuple = self.update_score(
                    score, encoder_outputs=encoder_outputs, encoder_mask=tf.to_float(
                        tf.not_equal(tf.stack(sentence_complex_input_placeholder, axis=1),
                                     self.data.vocab_complex.encode(constant.SYMBOL_PAD))),
                    comp_features=comp_features)

                encoder_outputs = self.update_encoder_embedding(encoder_outputs, score)

        encoder_embed_inputs_list = tf.unstack(encoder_embed_inputs, axis=1)

        with tf.variable_scope('transformer_decoder', reuse=tf.AUTO_REUSE):
            if self.model_config.subword_vocab_size or 'bert_token' in self.model_config.bert_mode:
                go_id = self.data.vocab_simple.encode(constant.SYMBOL_GO)[0]
            else:
                go_id = self.data.vocab_simple.encode(constant.SYMBOL_GO)
            batch_go = tf.tile(
                tf.expand_dims(self.embedding_fn(go_id, emb_simple), axis=0),
                [self.model_config.batch_size, 1])

            # For static_seq train_mode
            if self.model_config.npad_mode == 'static_seq':
                with tf.variable_scope('npad'):
                    npad_w = tf.get_variable(
                        'npad_w', shape=[1, self.model_config.dimension, self.model_config.dimension],
                        initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                    obj_tensors['npad_w'] = npad_w

            if self.is_train and (train_mode == 'teacher' or
                                  train_mode == 'teachercritical'or train_mode ==  'teachercriticalv2'):
                # General train
                print('Use Generally Process.')
                decoder_embed_inputs_list = self.embedding_fn(
                    sentence_simple_input_placeholder[:-1], emb_simple)
                final_output, decoder_output, cur_context = self.decode_step(
                    decoder_embed_inputs_list, encoder_outputs, encoder_attn_bias,
                    rule_id_input_placeholder, mem_contexts, mem_outputs, global_step, score, batch_go,
                    obj_tensors)

                decoder_logit = (
                        tf.nn.conv1d(final_output, tf.expand_dims(tf.transpose(w), axis=0), 1, 'SAME') +
                        tf.expand_dims(tf.expand_dims(b, axis=0), axis=0))
                decoder_target_list = []
                decoder_logit_list = tf.unstack(decoder_logit, axis=1)
                for logit in decoder_logit_list:
                    decoder_target_list.append(tf.argmax(logit, output_type=tf.int32, axis=-1))

                decoder_output_list = [
                    tf.squeeze(d, 1)
                    for d in tf.split(decoder_output, self.model_config.max_simple_sentence, axis=1)]
                final_output_list = [
                    tf.squeeze(d, 1)
                    for d in tf.split(final_output, self.model_config.max_simple_sentence, axis=1)]

                if self.model_config.pointer_mode:
                    segment_mask = None
                    if 'line_comp_segids' in obj:
                        segment_mask = obj['line_comp_segids']
                    decoder_logit_list = word_distribution(
                        decoder_logit_list, decoder_output_list, encoder_outputs, encoder_embed_inputs,
                        sentence_complex_input_placeholder, obj_tensors, self.model_config, self.data, segment_mask)
            elif self.is_train and (train_mode == 'static_seq' or train_mode == 'static_self-critical'):
                decoder_target_list = []
                decoder_logit_list = []
                decoder_embed_inputs_list = []
                # Will Override for following 3 lists
                final_output_list = []
                decoder_output_list = []
                contexts = []
                sample_target_list = []
                sample_logit_list = []

                gpu_assign_interval = int(self.model_config.max_simple_sentence / 3)
                for step in range(self.model_config.max_simple_sentence):
                    gpu_id = int(step / gpu_assign_interval)
                    if gpu_id > 3:
                        gpu_id = 3
                    gpu_id += 1
                    with tf.device('/device:GPU:%s' % gpu_id):
                        print('Step%s with GPU%s' % (step, gpu_id))
                        final_outputs, _, cur_context = self.decode_step(
                            decoder_embed_inputs_list, encoder_outputs, encoder_attn_bias,
                            rule_id_input_placeholder, mem_contexts, mem_outputs, global_step,
                            score, batch_go, obj_tensors)

                        final_output_list = [
                            tf.squeeze(d, 1)
                            for d in tf.split(final_outputs, step+1, axis=1)]
                        final_output = final_output_list[-1]

                        # if self.model_config.npad_mode == 'static_seq':
                        #     final_output = tf.matmul(final_output, npad_w)

                        last_logit_list = self.output_to_logit(final_output, w, b)
                        last_target_list = tf.argmax(last_logit_list, output_type=tf.int32, axis=-1)
                        decoder_logit_list.append(last_logit_list)
                        decoder_target_list.append(last_target_list)
                        decoder_embed_inputs_list.append(
                            tf.stop_gradient(self.embedding_fn(last_target_list, emb_simple)))
                        if train_mode == 'static_self-critical':
                            last_sample_list = tf.multinomial(last_logit_list, 1)
                            sample_target_list.append(last_sample_list)
                            indices = tf.stack(
                                [tf.range(0, self.model_config.batch_size, dtype=tf.int64),
                                 tf.squeeze(last_sample_list)],
                                axis=-1)
                            sample_logit_list.append(tf.gather_nd(tf.nn.softmax(last_logit_list), indices))
            else:
                # Beam Search
                print('Use Beam Search with Beam Search Size %d.' % self.model_config.beam_search_size)
                return self.transformer_beam_search(encoder_outputs, encoder_attn_bias, encoder_embed_inputs_list,
                                                    sentence_complex_input_placeholder, emb_simple, w, b,
                                                    rule_id_input_placeholder, mem_contexts, mem_outputs, global_step,
                                                    score, obj, obj_tensors)

        gt_target_list = sentence_simple_input_placeholder
        output = ModelOutput(
            contexts=cur_context if 'rule' in self.model_config.memory else None,
            encoder_outputs=encoder_outputs,
            decoder_outputs_list=final_output_list if train_mode != 'dynamic_self-critical' else None,
            final_outputs_list=final_output_list if train_mode != 'dynamic_self-critical' else None,
            decoder_logit_list=decoder_logit_list if train_mode != 'dynamic_self-critical' else None,
            gt_target_list=gt_target_list,
            encoder_embed_inputs_list=tf.unstack(encoder_embed_inputs, axis=1),
            decoder_target_list=decoder_target_list,
            sample_logit_list=sampled_logit_list if train_mode == 'dynamic_self-critical' else None,
            sample_target_list=sampled_target_list if train_mode == 'dynamic_self-critical' else None,
            pred_score_tuple=pred_score_tuple if 'pred' in self.model_config.tune_mode else None,
            obj_tensors=obj_tensors,
        )
        return output

    def decode_step(self, decode_input_list, encoder_outputs, encoder_attn_bias,
                    rule_id_input_placeholder, mem_contexts, mem_outputs, global_step, score, batch_go,
                    obj_tensors=None):
        # target_length = len(decode_input_list) + 1
        decoder_emb_inputs = tf.stack([batch_go] + decode_input_list, axis=1)
        final_output, decoder_output, cur_context = self.decode_inputs_to_outputs(
            decoder_emb_inputs, encoder_outputs, encoder_attn_bias,
            rule_id_input_placeholder, mem_contexts, mem_outputs, global_step, score,
            obj_tensors)

        # decoder_output_list = [
        #     tf.squeeze(d, 1)
        #     for d in tf.split(decoder_output, target_length, axis=1)]
        # final_output_list = [
        #     tf.squeeze(d, 1)
        #     for d in tf.split(final_output, target_length, axis=1)]
        return final_output, decoder_output, cur_context

    def transformer_beam_search(self, encoder_outputs, encoder_attn_bias, encoder_embed_inputs_list,
                                sentence_complex_input_placeholder, emb_simple, w, b,
                                rule_id_input_placeholder, mem_contexts, mem_outputs, global_step,
                                score, obj, obj_tensors):
        # Use Beam Search in evaluation stage
        # Update [a, b, c] to [a, a, a, b, b, b, c, c, c] if beam_search_size == 3
        encoder_beam_outputs = tf.concat(
            [tf.tile(tf.expand_dims(encoder_outputs[o, :, :], axis=0),
                     [self.model_config.beam_search_size, 1, 1])
             for o in range(self.model_config.batch_size)], axis=0)

        encoder_embed_inputs = tf.stack(encoder_embed_inputs_list, axis=1)
        encoder_beam_embed_inputs = tf.concat(
            [tf.tile(tf.expand_dims(encoder_embed_inputs[o, :, :], axis=0),
                     [self.model_config.beam_search_size, 1, 1])
             for o in range(self.model_config.batch_size)], axis=0)

        encoder_attn_beam_bias = tf.concat(
            [tf.tile(tf.expand_dims(encoder_attn_bias[o, :, :, :], axis=0),
                     [self.model_config.beam_search_size, 1, 1, 1])
             for o in range(self.model_config.batch_size)], axis=0)

        if 'direct' in self.model_config.memory:
            obj_tensors['direct_bert_output_bak'] = obj_tensors['direct_bert_output']
            obj_tensors['direct_bert_bias_bak'] = obj_tensors['direct_bert_bias']
            obj_tensors['direct_bert_output'] = tf.concat(
                [tf.tile(tf.expand_dims(obj_tensors['direct_bert_output'][o, :, :], axis=0),
                         [self.model_config.beam_search_size, 1, 1])
                 for o in range(self.model_config.batch_size)], axis=0)
            obj_tensors['direct_bert_bias'] = tf.concat(
                [tf.tile(tf.expand_dims(obj_tensors['direct_bert_bias'][o, :, :, :], axis=0),
                         [self.model_config.beam_search_size, 1, 1, 1])
                 for o in range(self.model_config.batch_size)], axis=0)

        if self.model_config.subword_vocab_size or 'bert_token' in self.model_config.bert_mode:
            go_id = self.data.vocab_simple.encode(constant.SYMBOL_GO)[0]
            eos_id = self.data.vocab_simple.encode(constant.SYMBOL_END)[0]
        else:
            go_id = self.data.vocab_simple.encode(constant.SYMBOL_GO)
            eos_id = self.data.vocab_simple.encode(constant.SYMBOL_END)
        batch_go = tf.expand_dims(tf.tile(
            tf.expand_dims(self.embedding_fn(go_id, emb_simple), axis=0),
            [self.model_config.batch_size, 1]), axis=1)
        batch_go_beam = tf.concat(
            [tf.tile(tf.expand_dims(batch_go[o, :, :], axis=0),
                     [self.model_config.beam_search_size, 1, 1])
             for o in range(self.model_config.batch_size)], axis=0)

        def symbol_to_logits_fn(ids):
            cur_ids = ids[:, 1:]
            embs = tf.nn.embedding_lookup(emb_simple, cur_ids)
            embs = tf.concat([batch_go_beam, embs], axis=1)

            final_outputs, _, _ = self.decode_inputs_to_outputs(embs, encoder_beam_outputs, encoder_attn_beam_bias,
                                                                rule_id_input_placeholder, mem_contexts, mem_outputs,
                                                                global_step, score, obj_tensors=obj_tensors)

            decoder_logit_list = self.output_to_logit(final_outputs[:, -1, :], w, b)

            if self.model_config.pointer_mode:
                segment_mask = None
                if 'line_comp_segids' in obj:
                    segment_mask = obj['line_comp_segids']
                decoder_logit_list = word_distribution(
                    [decoder_logit_list], [final_outputs[:, -1, :]],
                    encoder_beam_outputs, encoder_beam_embed_inputs,
                    sentence_complex_input_placeholder,
                    obj_tensors, self.model_config, self.data, segment_mask, is_test=True)

            return decoder_logit_list

        beam_ids, beam_score = beam_search.beam_search(symbol_to_logits_fn,
                                                       tf.ones([self.model_config.batch_size], tf.int32) * go_id,
                                                       self.model_config.beam_search_size,
                                                       self.model_config.max_simple_sentence,
                                                       self.data.vocab_simple.vocab_size(),
                                                       self.model_config.penalty_alpha,
                                                       eos_id=eos_id
                                                       )
        top_beam_ids = beam_ids[:, 0, 1:]
        top_beam_ids = tf.pad(top_beam_ids,
                              [[0, 0],
                               [0, self.model_config.max_simple_sentence - tf.shape(top_beam_ids)[1]]])

        decoder_target_list = [tf.squeeze(d, 1)
                               for d in tf.split(top_beam_ids, self.model_config.max_simple_sentence, axis=1)]
        decoder_score = -beam_score[:, 0] / tf.to_float(tf.shape(top_beam_ids)[1])

        # Get outputs based on target ids
        decode_input_embs = tf.stack(self.embedding_fn(decoder_target_list, emb_simple), axis=1)
        tf.get_variable_scope().reuse_variables()
        if 'direct' in self.model_config.memory:
            obj_tensors['direct_bert_output'] = obj_tensors['direct_bert_output_bak']
            obj_tensors['direct_bert_bias'] = obj_tensors['direct_bert_bias_bak']
        final_outputs, decoder_outputs, _ = self.decode_inputs_to_outputs(decode_input_embs, encoder_outputs, encoder_attn_bias,
                                                                          rule_id_input_placeholder, mem_contexts,
                                                                          mem_outputs, global_step, score,
                                                                          obj_tensors=obj_tensors)
        output = ModelOutput(
            encoder_outputs=encoder_outputs,
            final_outputs_list=final_outputs,
            decoder_outputs_list=decoder_outputs,
            decoder_score=decoder_score,
            decoder_target_list=decoder_target_list,
            encoder_embed_inputs_list=encoder_embed_inputs_list
        )
        return output

    def output_to_logit(self, prev_out, w, b):
        prev_logit = tf.add(tf.matmul(prev_out, tf.transpose(w)), b)
        return prev_logit

    def decode_inputs_to_outputs(self, decoder_embed_inputs, encoder_outputs, encoder_attn_bias,
                                 rule_id_input_placeholder, mem_contexts, mem_outputs, global_step,
                                 score, obj_tensors=None):
        if self.hparams.pos == 'timing':
            decoder_embed_inputs = common_attention.add_timing_signal_1d(decoder_embed_inputs)
            print('Use positional encoding in decoder text.')
        decoder_embed_inputs = self.update_decoder_embedding(decoder_embed_inputs, score, self.model_config.beam_search_size)

        decoder_attn_bias = common_attention.attention_bias_lower_triangle(tf.shape(decoder_embed_inputs)[1])
        decoder_embed_inputs = tf.nn.dropout(decoder_embed_inputs,
                                             1.0 - self.hparams.layer_prepostprocess_dropout)
        if 'direct' in self.model_config.memory:
            assert 'direct_bert_output' in obj_tensors
            decoder_output = transformer.transformer_multi_decoder(
                decoder_embed_inputs, encoder_outputs, decoder_attn_bias,
                encoder_attn_bias, obj_tensors['direct_bert_output'], obj_tensors['direct_bert_bias'],
                self.hparams, save_weights_to=obj_tensors,
                direct_mode=self.model_config.direct_mode)

            if self.model_config.npad_mode == 'static_seq':
                decoder_output = tf.nn.conv1d(decoder_output, obj_tensors['npad_w'], 1, 'SAME')

            return decoder_output, decoder_output, None
        elif 'rule' in self.model_config.memory:
            decoder_output, contexts = transformer.transformer_decoder_contexts(
                decoder_embed_inputs, encoder_outputs, decoder_attn_bias,
                encoder_attn_bias, self.hparams)

            # encoder_gate_w = tf.get_variable('encoder_gate_w', shape=(
            #     1, self.model_config.dimension, 1))
            # encoder_gate_b = tf.get_variable('encoder_gate_b', shape=(1, 1, 1))
            # encoder_gate = tf.tanh(encoder_gate_b + tf.nn.conv1d(encoder_outputs, encoder_gate_w, 1, 'SAME'))
            # encoder_context_outputs = tf.expand_dims(tf.reduce_mean(encoder_outputs * encoder_gate, axis=1), axis=1)
            cur_context = contexts[0] #tf.concat(contexts, axis=-1)
            cur_mem_contexts = tf.stack(self.embedding_fn(rule_id_input_placeholder, mem_contexts), axis=1)
            cur_mem_outputs = tf.stack(self.embedding_fn(rule_id_input_placeholder, mem_outputs), axis=1)
            cur_mem_contexts = tf.reshape(cur_mem_contexts,
                                          [self.model_config.batch_size,
                                           self.model_config.max_target_rule_sublen*self.model_config.max_cand_rules,
                                           self.model_config.dimension])
            cur_mem_outputs = tf.reshape(cur_mem_outputs,
                                         [self.model_config.batch_size,
                                          self.model_config.max_target_rule_sublen*self.model_config.max_cand_rules,
                                          self.model_config.dimension])

            # bias = tf.expand_dims(
            #     -1e9 * tf.to_float(tf.equal(tf.stack(rule_id_input_placeholder, axis=1), 0)),
            #     axis=1)
            # weights = tf.nn.softmax(bias + tf.matmul(cur_context, cur_mem_contexts, transpose_b=True))
            weights = tf.nn.softmax(tf.matmul(cur_context, cur_mem_contexts, transpose_b=True))
            mem_output = tf.matmul(weights, cur_mem_outputs)

            # trainable_mem = 'stopgrad' not in self.model_config.rl_configs
            temp_output = tf.concat((decoder_output, mem_output), axis=-1)
            # w_u = tf.get_variable('w_ffn', shape=(
            #     1, self.model_config.dimension*2, self.model_config.dimension), trainable=trainable_mem)
            # b_u = tf.get_variable('b_ffn', shape=(
            #     1, 1, self.model_config.dimension), trainable=trainable_mem)
            # w_u.reuse_variables()
            # b_u.reuse_variables()
            # tf.get_variable_scope().reuse_variables()
            w_t = tf.get_variable('w_ffn', shape=(
                1, self.model_config.dimension*2, self.model_config.dimension), trainable=True)
            b_t = tf.get_variable('b_ffn', shape=(
                1, 1, self.model_config.dimension), trainable=True)
            # w = tf.cond(tf.equal(tf.mod(self.global_step, 2), 0), lambda: w_t, lambda: w_u)
            # b = tf.cond(tf.equal(tf.mod(self.global_step, 2), 0), lambda: b_t, lambda: b_u)

            mem_output = tf.nn.conv1d(temp_output, w_t, 1, 'SAME') + b_t
            g = tf.greater(global_step, tf.constant(self.model_config.memory_prepare_step, dtype=tf.int64))
            final_output = tf.cond(g, lambda: mem_output, lambda: decoder_output)
            return final_output, decoder_output, cur_context
        else:
            if self.model_config.architecture == 'ut2t':
                (decoder_output, decoder_extra_output) = universal_transformer_util.universal_transformer_decoder(
                    decoder_embed_inputs, encoder_outputs,
                    decoder_attn_bias, encoder_attn_bias, self.hparams,
                    save_weights_to=obj_tensors)
                dec_ponder_times, dec_remainders = decoder_extra_output
                extra_dec_loss = (
                        self.hparams.act_loss_weight *
                        tf.reduce_mean(dec_ponder_times + dec_remainders))
                if self.is_train:
                    obj_tensors['extra_decoder_loss'] = extra_dec_loss
            else:
                decoder_output = transformer.transformer_decoder(
                    decoder_embed_inputs, encoder_outputs, decoder_attn_bias,
                    encoder_attn_bias, self.hparams, save_weights_to=obj_tensors,
                    npad_mode=self.model_config.npad_mode)
            final_output = decoder_output
            return final_output, decoder_output, None

    def setup_hparams(self):
        self.hparams.num_heads = self.model_config.num_heads
        self.hparams.num_hidden_layers = self.model_config.num_hidden_layers
        self.hparams.num_encoder_layers = self.model_config.num_encoder_layers
        self.hparams.num_decoder_layers = self.model_config.num_decoder_layers
        self.hparams.pos = self.model_config.hparams_pos
        self.hparams.hidden_size = self.model_config.dimension
        self.hparams.layer_prepostprocess_dropout = self.model_config.layer_prepostprocess_dropout

        if self.model_config.architecture == 'ut2t':
            self.hparams = universal_transformer.update_hparams_for_universal_transformer(self.hparams)
            self.hparams.recurrence_type = "act"

        if self.is_train:
            self.hparams.add_hparam('mode', tf.estimator.ModeKeys.TRAIN)
        else:
            self.hparams.add_hparam('mode', tf.estimator.ModeKeys.EVAL)
            self.hparams.layer_prepostprocess_dropout = 0.0
            self.hparams.attention_dropout = 0.0
            self.hparams.dropout = 0.0
            self.hparams.relu_dropout = 0.0

