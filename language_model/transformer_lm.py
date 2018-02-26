"""
Transformer-based LM: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/attention_lm.py
"""
import tensorflow as tf
from tensorflow.contrib.seq2seq import sequence_loss

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer

from util import constant
from language_model.lm_arguments import get_args

args = get_args()


class TransformerLM:
    def setup_hparams(self, hparams):
        hparams.num_heads = args.num_heads
        hparams.num_hidden_layers = args.num_hidden_layers
        hparams.num_decoder_layers = args.num_hidden_layers
        hparams.hidden_size = args.dimension
        hparams.layer_prepostprocess_dropout = args.layer_prepostprocess_dropout
        return hparams

    def create_model_multigpu(self, data, is_train):
        losses = []
        grads = []
        self.objs = []
        self.global_step = tf.get_variable(
            'global_step', initializer=tf.constant(0, dtype=tf.int64), trainable=False)
        optim = tf.train.AdagradOptimizer(args.learning_rate)

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for gpu_id in range(args.num_gpus):
                with tf.device('/gpu:%d' % gpu_id):
                    loss, obj = self.create_model(data)
                    self.objs.append(obj)
                    grad = optim.compute_gradients(loss)
                    losses.append(loss)
                    grads.append(grad)
                    tf.get_variable_scope().reuse_variables()
        self.loss = tf.divide(tf.add_n(losses), args.num_gpus)
        self.perplexity = tf.exp(tf.reduce_mean(self.loss))

        if is_train:
            avg_grad = self.average_gradients(grads)
            grads = [g for (g, v) in avg_grad]
            clipped_grads, _ = tf.clip_by_global_norm(grads, 5.0)
            self.train_op = optim.apply_gradients(zip(clipped_grads, tf.trainable_variables()),
                                                  global_step=self.global_step)
            self.increment_global_step = tf.assign_add(self.global_step, 1)

        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

    def create_model(self, data):
        with tf.variable_scope('embedding'):
            with tf.device('/cpu:0'):
                embedding = tf.get_variable(
                    'embedding', [data.vocab.vocab_size(), args.dimension], tf.float32,
                    initializer=tf.random_uniform_initializer(-0.08, 0.08))

            proj_w = tf.get_variable(
                'output_w', [data.vocab.vocab_size(), args.dimension], tf.float32,
                initializer=tf.random_uniform_initializer(-0.08, 0.08))
            proj_b = tf.get_variable(
                'output_b', shape=[data.vocab.vocab_size()],
                initializer=tf.random_uniform_initializer(-0.08, 0.08))

        with tf.variable_scope('variables'):
            sentence_inputs = []
            for step in range(args.max_sent_len):
                sentence_inputs.append(
                    tf.zeros(args.batch_size, tf.int32, name='input'))

            sentence_outputs_list = [tf.constant(data.vocab.encode(constant.SYMBOL_GO), shape=sentence_inputs[0].get_shape())] +\
                                    sentence_inputs[1:]
            sentence_outputs = tf.stack(sentence_outputs_list, axis=1)
            sentence_inputs_emb = tf.stack(self.embedding_fn(sentence_inputs, embedding), axis=1)

        hparams = transformer.transformer_base()
        hparams = self.setup_hparams(hparams)
        (decoder_input, decoder_self_attention_bias) = self.attention_lm_prepare_decoder(
            sentence_inputs_emb, hparams)

        decoder_input = tf.nn.dropout(decoder_input,
                                      1.0 - hparams.layer_prepostprocess_dropout)
        decoder_outputs = self.attention_lm_decoder(decoder_input,
                                              decoder_self_attention_bias, hparams)
        decoder_outputs_list = [tf.squeeze(d, 1)
            for d in tf.split(decoder_outputs, args.max_sent_len, axis=1)]
        decoder_logits_list = [tf.add(tf.matmul(decoder_output, tf.transpose(proj_w)), proj_b) for decoder_output in decoder_outputs_list]
        decoder_logits = tf.stack(decoder_logits_list, axis=1)

        decode_word_weight = tf.stack(
            [tf.to_float(tf.not_equal(d, data.vocab.encode(constant.SYMBOL_PAD)))
             for d in sentence_outputs_list], axis=1)

        loss = sequence_loss(decoder_logits, sentence_outputs, decode_word_weight)
        obj = {
            'sentence_inputs':sentence_inputs,
        }
        return loss, obj

    def attention_lm_prepare_decoder(self, targets, hparams):
        """Prepare one shard of the model for the decoder.
        Args:
          targets: a Tensor.
          hparams: run hyperparameters
        Returns:
          decoder_input: a Tensor, bottom of decoder stack
          decoder_self_attention_bias: a Tensor, containing large negative values
          to implement masked attention and possibly baises for diagonal alignments
        """
        decoder_self_attention_bias = (
                common_attention.attention_bias_lower_triangle(
                    common_layers.shape_list(targets)[1]))
        decoder_input = common_layers.shift_right_3d(targets)
        if hparams.pos == "timing":
            decoder_input = common_attention.add_timing_signal_1d(decoder_input)
        return (decoder_input, decoder_self_attention_bias)

    def attention_lm_decoder(self,
                             decoder_input,
                             decoder_self_attention_bias,
                             hparams,
                             name="decoder"):
      """A stack of attention_lm layers.
      Args:
        decoder_input: a Tensor
        decoder_self_attention_bias: bias Tensor for self-attention
          (see common_attention.attention_bias())
        hparams: hyperparameters for model
        name: a string
      Returns:
        y: a Tensors
      """
      x = decoder_input
      with tf.variable_scope(name):
        for layer in range(hparams.num_hidden_layers):
          with tf.variable_scope("layer_%d" % layer):
            with tf.variable_scope("self_attention"):
              y = common_attention.multihead_attention(
                  common_layers.layer_preprocess(
                      x, hparams), None, decoder_self_attention_bias,
                  hparams.attention_key_channels or hparams.hidden_size,
                  hparams.attention_value_channels or hparams.hidden_size,
                  hparams.hidden_size, hparams.num_heads, hparams.attention_dropout)
              x = common_layers.layer_postprocess(x, y, hparams)
            with tf.variable_scope("ffn"):
              y = common_layers.conv_hidden_relu(
                  common_layers.layer_preprocess(x, hparams),
                  hparams.filter_size,
                  hparams.hidden_size,
                  dropout=hparams.relu_dropout)
              x = common_layers.layer_postprocess(x, y, hparams)
        return common_layers.layer_preprocess(x, hparams)

    def embedding_fn(self, inputs, embedding):
      # with tf.device(self.device_config):
      if not inputs:
          return []
      else:
          return [tf.nn.embedding_lookup(embedding, inp) for inp in inputs]  # Got from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L101  # Got from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L101

    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads