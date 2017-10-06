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

    def embedding_fn(self, inputs, embedding):
        if not inputs:
            return []
        else:
            return [tf.nn.embedding_lookup(embedding, inp) for inp in inputs]

    def create_model(self):
        with tf.variable_scope('variables'):
            self.sentence_simple_input_placeholder = []
            for step in range(self.model_config.max_simple_sentence):
                self.sentence_simple_input_placeholder.append(tf.zeros(self.model_config.batch_size,
                                                                       tf.int32, name='simple_input'))

            self.sentence_complex_input_placeholder = []
            for step in range(self.model_config.max_complex_sentence):
                self.sentence_complex_input_placeholder.append(tf.zeros(self.model_config.batch_size,
                                                                        tf.int32, name='complex_input'))

            self.embedding = Embedding(self.data.vocab_complex, self.data.vocab_simple, self.model_config)
            self.emb_complex = self.embedding.get_complex_embedding()
            self.emb_simple = self.embedding.get_simple_embedding()
            if self.is_train and self.model_config.pretrained_embedding is not None:
                self.embed_complex_placeholder = tf.placeholder(
                    tf.float32, (len(self.data.vocab_complex.i2w), self.model_config.dimension),
                    'complex_emb')
                self.replace_emb_complex = self.emb_complex.assign(self.embed_complex_placeholder)

                self.embed_simple_placeholder = tf.placeholder(
                    tf.float32, (len(self.data.vocab_simple.i2w), self.model_config.dimension),
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
                self.encoder_embs = tf.stack(
                    self.embedding_fn(self.sentence_complex_input_placeholder, self.emb_complex),
                    axis=1)
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

                if self.model_config.use_quality_model:
                    metric = Metric(self.model_config, self.data)
                    sentence_simple_input_mat = tf.stack(self.sentence_simple_input_placeholder, axis=1)
                    sentence_complex_input_mat = tf.stack(self.sentence_complex_input_placeholder, axis=1)
                    weight_quality = tf.py_func(metric.length_ratio,
                                                [sentence_simple_input_mat,
                                                 sentence_complex_input_mat],
                                                [tf.float32], stateful=False, name='quality_weight')
                    weight_quality[0].set_shape([self.model_config.batch_size])
                    weight_quality = tf.stack(
                        [weight_quality[0] for _ in range(self.model_config.max_complex_sentence)], axis=-1)

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
                #                 num_classes=len(self.data.vocab_simple.i2w))
                #
                #     loss_fn = _sampled_softmax


                self.loss = sequence_loss(logits=tf.stack(output.decoder_logit_list, axis=1),
                                          targets=tf.stack(output.gt_target_list, axis=1),
                                          weights=decode_word_weight,
                                          softmax_loss_function=loss_fn)


        with tf.variable_scope('optimization'):
            self.global_step = tf.get_variable(
                'global_step', initializer=tf.constant(0, dtype=tf.int64), trainable=False)

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
        elif self.model_config.optimizer == 'adam_transformer_simple':
            if not hasattr(self, 'hparams'):
                # In case not using Transformer model
                from tensor2tensor.models import transformer
                self.hparams = transformer.transformer_base()
            opt = tf.contrib.opt.LazyAdamOptimizer(
                self.hparams.learning_rate / 500.0,
                beta1=self.hparams.optimizer_adam_beta1,
                beta2=self.hparams.optimizer_adam_beta2,
                epsilon=self.hparams.optimizer_adam_epsilon)
        elif self.model_config.optimizer == 'adam_transformer':
            if not hasattr(self, 'hparams'):
                # In case not using Transformer model
                from tensor2tensor.models import transformer
                self.hparams = transformer.transformer_base()
            return TransformerOptimizer(self.loss, self.hparams, self.global_step).get_op()
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
                 decoder_score=None, gt_target_list=None):
        self._decoder_outputs = decoder_outputs
        self._decoder_logit_list = decoder_logit_list
        self._decoder_target_list = decoder_target_list
        self._decoder_score = decoder_score
        self._gt_target_list = gt_target_list

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
