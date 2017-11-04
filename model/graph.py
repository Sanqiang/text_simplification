from model.embedding import Embedding
from model.loss import sequence_loss
from model.metric import Metric
from model.optimizer import TransformerOptimizer
from util import constant

import tensorflow as tf


class Graph:
    def __init__(self, data, is_train, model_config):
        self.model_config = model_config
        self.data = data
        self.is_train = is_train
        # model_fn defines core computational graph
        # decoder_outputs, logits, target_outputs
        # decoder_outputs is [batch * length * dimension]
        # logits is [batch * length * vocab_size]
        # target_outputs is [batch * length * vocab_size]
        # in training, target_outputs is gt target
        # in eval, target_outputs is output target
        self.model_fn = None
        print('Batch Size:\t%d.' % self.model_config.batch_size)
        self.rand_unif_init = tf.random_uniform_initializer(-0,.08, 0.08)
        self.metric = Metric(self.model_config, self.data)

        # self.device_config = '/gpu:0'
        # if self.model_config.use_cpu:
        #     self.device_config = '/cpu:0'

    def embedding_fn(self, inputs, embedding):
        # with tf.device(self.device_config):
        if not inputs:
            return []
        else:
            return [tf.nn.embedding_lookup(embedding, inp) for inp in inputs]

    def create_model(self):
        with tf.variable_scope('variables'):
            self.sentence_simple_input_placeholder = []
            for step in range(self.model_config.max_simple_sentence):
                self.sentence_simple_input_placeholder.append(
                    tf.zeros(self.model_config.batch_size, tf.int32, name='simple_input'))

            self.sentence_simple_input_prior_placeholder = []
            for step in range(self.model_config.max_simple_sentence):
                self.sentence_simple_input_prior_placeholder.append(
                    tf.ones(self.model_config.batch_size, tf.float32, name='simple_input_prior'))

            self.sentence_complex_input_placeholder = []
            for step in range(self.model_config.max_complex_sentence):
                self.sentence_complex_input_placeholder.append(
                    tf.zeros(self.model_config.batch_size, tf.int32, name='complex_input'))

            self.sentence_complex_attn_prior_input_placeholder = []
            for step in range(self.model_config.max_complex_sentence):
                self.sentence_complex_attn_prior_input_placeholder.append(
                    tf.ones(self.model_config.batch_size, tf.float32, name='complex_input'))

            self.embedding = Embedding(self.data.vocab_complex, self.data.vocab_simple, self.model_config)
            # with tf.device(self.device_config):
            self.emb_complex = self.embedding.get_complex_embedding()
            self.emb_simple = self.embedding.get_simple_embedding()
            if (self.is_train and self.model_config.pretrained_embedding is not None and
                        self.model_config.subword_vocab_size <= 0):
                self.embed_complex_placeholder = tf.placeholder(
                    tf.float32, (self.data.vocab_complex.vocab_size(), self.model_config.dimension),
                    'complex_emb')
                self.replace_emb_complex = self.emb_complex.assign(self.embed_complex_placeholder)

                self.embed_simple_placeholder = tf.placeholder(
                    tf.float32, (self.data.vocab_simple.vocab_size(), self.model_config.dimension),
                    'simple_emb')
                self.replace_emb_simple = self.emb_simple.assign(self.embed_simple_placeholder)

            self.w = self.embedding.get_w()
            self.b = self.embedding.get_b()

        with tf.variable_scope('model'):
            output = self.model_fn()

            if output.decoder_target_list is None:
                # For train or model_fn doesn't provide decoder target list
                # Get decode target list based on decoder logit list
                self.decoder_target_list = tf.stack(
                    [tf.argmax(logit, axis=1) for logit in output.decoder_logit_list],
                    axis=1)

            if not self.is_train and self.model_config.replace_unk_by_emb:
                # Get output list matrix for replacement by embedding
                # self.encoder_embs = tf.stack(
                #     self.embedding_fn(self.sentence_complex_input_placeholder, self.emb_complex),
                #     axis=1)
                self.encoder_embs = tf.stack(output.encoder_embed_inputs_list, axis=1)
                if type(output.decoder_outputs) == list:
                    self.decoder_outputs = tf.stack(output.decoder_outputs, axis=1)
                else:
                    self.decoder_outputs = output.decoder_outputs

            if (not self.is_train and self.model_config.beam_search_size > 0 and
                        self.model_config.framework == 'transformer'):
                # in beam search, it directly provide decoder target list
                self.decoder_target_list = tf.stack(output.decoder_target_list, axis=1)
                self.loss = tf.reduce_mean(output.decoder_score)
            else:
                decode_word_weight = tf.stack(
                    [tf.to_float(tf.not_equal(d, self.data.vocab_simple.encode(constant.SYMBOL_PAD)))
                     for d in output.gt_target_list], axis=1)

                prior_weight = tf.stack(self.sentence_simple_input_prior_placeholder, axis=1)
                decode_word_weight = tf.multiply(prior_weight, decode_word_weight)

                # Get gt_target (either original one or from rl process)
                if self.is_train and (
                                self.model_config.rl_bleu or
                                self.model_config.rl_sari or
                            self.model_config.rl_fkgl):
                    # Use RL
                    decoder_word_list = [tf.argmax(logit, axis=-1)
                                         for logit in output.decoder_logit_list]
                    gt_target, weight_rl = tf.py_func(self.metric.rl_process,
                                                           [
                                                               tf.stack(self.sentence_complex_input_placeholder, axis=1),
                                                               tf.stack(self.sentence_simple_input_placeholder, axis=1),
                                                               tf.stack(decoder_word_list, axis=1)
                                                            ],
                                                           [tf.int32, tf.float32],
                                                           stateful=False,
                                                           name='rl_process')
                    gt_target.set_shape([self.model_config.batch_size, self.model_config.max_simple_sentence])
                    weight_rl.set_shape([self.model_config.batch_size, self.model_config.max_simple_sentence])
                    decode_word_weight = weight_rl

                else:
                    gt_target = tf.stack(output.gt_target_list, axis=1)

                if self.model_config.use_quality_model:
                    sentence_simple_input_mat = tf.stack(self.sentence_simple_input_placeholder, axis=1)
                    sentence_complex_input_mat = tf.stack(self.sentence_complex_input_placeholder, axis=1)
                    weight_quality = tf.py_func(self.metric.lm_quality,
                                                [
                                                    sentence_simple_input_mat,
                                                    # sentence_complex_input_mat
                                                 ],
                                                tf.float32, stateful=False, name='quality_weight')
                    weight_quality.set_shape([self.model_config.batch_size])
                    weight_quality =tf.expand_dims(weight_quality, axis=-1)
                    # weight_quality = tf.stack(
                    #     [weight_quality[0] for _ in range(self.model_config.max_simple_sentence)], axis=-1)

                    decode_word_weight = tf.multiply(decode_word_weight, weight_quality)

                loss_fn = None
                # if self.model_config.loss_fn == 'sampled_softmax':
                #     def _sampled_softmax(labels, inputs):
                #         labels = tf.reshape(labels, [-1, 1])
                #         # We need to compute the sampled_softmax_loss using 32bit floats to
                #         # avoid numerical instabilities.
                #         local_w_t = tf.cast(self.w, tf.float32)
                #         local_b = tf.cast(self.b, tf.float32)
                #         local_inputs = tf.cast(inputs, tf.float32)
                #         return tf.nn.sampled_softmax_loss(
                #                 weights=local_w_t,
                #                 biases=local_b,
                #                 labels=labels,
                #                 inputs=local_inputs,
                #                 num_sampled=10,
                #                 num_classes=self.data.vocab_simple.vocab_size())
                #
                #     loss_fn = _sampled_softmax


                self.loss = sequence_loss(logits=tf.stack(output.decoder_logit_list, axis=1),
                                          targets=gt_target,
                                          weights=decode_word_weight,
                                          softmax_loss_function=loss_fn)

                if self.model_config.attn_loss:
                    if 'enc_self' in self.model_config.attn_loss:
                        for layer_i in range(self.model_config.num_encoder_layers):
                            att_dists = tf.get_default_graph().get_operation_by_name(
                                'model/transformer_encoder/encoder/layer_%s/self_attention/multihead_attention/dot_product_attention/attention_weights'
                                % layer_i).values()[0]
                            if self.model_config.ppdb_args and len(self.model_config.ppdb_args) >= 3:
                                self.model_config.attn_loss['enc_self'] *= self.model_config.ppdb_args[2]
                                attn_prior_input_placeholder = tf.expand_dims(tf.stack(
                                    self.sentence_complex_attn_prior_input_placeholder, axis=1), axis=1)
                                att_dists_sum = tf.reduce_sum(att_dists, axis=-1)
                                att_dists_sum *= attn_prior_input_placeholder
                                print('Use PPDB Attention Loss for enc_self.')
                            else:
                                att_dists_sum = tf.reduce_sum(att_dists, axis=-1)
                            att_dists_target = tf.ones(tf.shape(att_dists_sum))
                            att_loss = tf.losses.mean_squared_error(att_dists_target, att_dists_sum) * self.model_config.attn_loss['enc_self']
                            self.loss = tf.add(self.loss, att_loss)
                        print('Add Attention Reconstruction loss for enc_self with weight %s.'
                              % self.model_config.attn_loss['enc_self'])

                    if 'enc_dec' in self.model_config.attn_loss:
                        for layer_i in range(self.model_config.num_encoder_layers):
                            att_dists = tf.get_default_graph().get_operation_by_name(
                                'model/transformer_decoder/decoder/layer_%s/encdec_attention/multihead_attention/dot_product_attention/attention_weights'
                                % layer_i).values()[0]
                            if self.model_config.ppdb_args and len(self.model_config.ppdb_args) >= 3:
                                self.model_config.attn_loss['enc_dec'] *= self.model_config.ppdb_args[2]
                                attn_prior_input_placeholder = tf.expand_dims(tf.stack(
                                    self.sentence_complex_attn_prior_input_placeholder, axis=1), axis=1)
                                att_dists_sum = tf.reduce_sum(att_dists, axis=-1)
                                att_dists_sum *= attn_prior_input_placeholder
                                print('Use PPDB Attention Loss for enc_dec.')
                            else:
                                att_dists_sum = tf.reduce_sum(att_dists, axis=-1)
                            att_dists_target = tf.ones(tf.shape(att_dists_sum))
                            att_loss = tf.losses.mean_squared_error(att_dists_target, att_dists_sum) * self.model_config.attn_loss['enc_dec']
                            self.loss = tf.add(self.loss, att_loss)
                        print('Add Attention Reconstruction loss for enc_dec with weight %s.'
                              % self.model_config.attn_loss['enc_dec'])

                    if 'dec_self' in self.model_config.attn_loss:
                        for layer_i in range(self.model_config.num_encoder_layers):
                            att_dists = tf.get_default_graph().get_operation_by_name(
                                'model/transformer_decoder/decoder/layer_%s/self_attention/multihead_attention/dot_product_attention/attention_weights'
                                % layer_i).values()[0]
                            att_dists_sum = tf.reduce_sum(att_dists, axis=-1)
                            att_dists_target = tf.ones(tf.shape(att_dists_sum))
                            att_loss = tf.losses.mean_squared_error(att_dists_target, att_dists_sum) * self.model_config.attn_loss['dec_self']
                            self.loss = tf.add(self.loss, att_loss)
                        print('Add Attention Reconstruction loss for dec_self with weight %s.'
                              % self.model_config.attn_loss['dec_self'])


        with tf.variable_scope('optimization'):
            self.global_step = tf.get_variable(
                'global_step', initializer=tf.constant(0, dtype=tf.int64), trainable=False)

            if self.is_train:
                self.increment_global_step = tf.assign_add(self.global_step, 1)
                self.train_op = self.create_train_op()

            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        print('Graph Built.')

    def create_train_op(self):
        def learning_rate_decay(model_config, step, perplexity):
            learning_rate = tf.case({
                tf.less(perplexity, 10):
                    lambda : 0.0001,
                tf.logical_and(tf.less(perplexity, 50), tf.greater_equal(perplexity, 10)):
                    lambda: 0.0002,
                tf.logical_and(tf.less(perplexity, 100), tf.greater_equal(perplexity, 50)):
                    lambda: 0.0003,
                tf.logical_and(tf.less(perplexity, 500), tf.greater_equal(perplexity, 100)):
                    lambda: 0.0005,
            }, default=lambda : 0.001, exclusive=True)
            return learning_rate

        self.perplexity = tf.exp(tf.reduce_mean(self.loss))
        learning_rate = self.model_config.learning_rate
        if self.model_config.use_learning_rate_decay:
            learning_rate = learning_rate_decay(
                self.model_config, self.global_step, self.perplexity)
        self.learning_rate = tf.constant(learning_rate)

        if self.model_config.optimizer == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate)
        # Adam need lower learning rate
        elif self.model_config.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate)
        elif self.model_config.optimizer == 'lazy_adam':
            if not hasattr(self, 'hparams'):
                # In case not using Transformer model
                from tensor2tensor.models import transformer
                self.hparams = transformer.transformer_base()
            opt = tf.contrib.opt.LazyAdamOptimizer(
                self.hparams.learning_rate / 100.0,
                beta1=self.hparams.optimizer_adam_beta1,
                beta2=self.hparams.optimizer_adam_beta2,
                epsilon=self.hparams.optimizer_adam_epsilon)
        elif self.model_config.optimizer == 'adam_transformer':
            if not hasattr(self, 'hparams'):
                # In case not using Transformer model
                from tensor2tensor.models import transformer
                self.hparams = transformer.transformer_base()
            return TransformerOptimizer(self.loss, self.hparams, self.global_step).get_op()
        elif self.model_config.optimizer == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(learning_rate)
        elif self.model_config.optimizer == 'adagraddao':
            opt = tf.train.AdagradDAOptimizer(learning_rate, self.global_step)
        else:
            raise Exception('Not Implemented Optimizer!')

        if self.model_config.max_grad_staleness > 0:
            opt = tf.contrib.opt.DropStaleGradientOptimizer(opt, self.model_config.max_grad_staleness)

        grads_and_vars = opt.compute_gradients(self.loss, var_list=tf.trainable_variables())
        grads = [g for (g,v) in grads_and_vars]
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.model_config.max_grad_norm)

        return opt.apply_gradients(zip(clipped_grads, tf.trainable_variables()), global_step=self.global_step)


class ModelOutput:
    def __init__(self, decoder_outputs=None, decoder_logit_list=None, decoder_target_list=None,
                 decoder_score=None, gt_target_list=None, encoder_embed_inputs_list=None):
        self._decoder_outputs = decoder_outputs
        self._decoder_logit_list = decoder_logit_list
        self._decoder_target_list = decoder_target_list
        self._decoder_score = decoder_score
        self._gt_target_list = gt_target_list
        self._encoder_embed_inputs_list = encoder_embed_inputs_list

    @property
    def encoder_embed_inputs_list(self):
        """The final embedding input before model."""
        return self._encoder_embed_inputs_list

    @property
    def decoder_outputs(self):
        return self._decoder_outputs

    @property
    def decoder_logit_list(self):
        return self._decoder_logit_list

    @property
    def decoder_target_list(self):
        return self._decoder_target_list

    @property
    def decoder_score(self):
        return self._decoder_score

    @property
    def gt_target_list(self):
        return self._gt_target_list
