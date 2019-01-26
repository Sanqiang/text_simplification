"""Transformer Optimization Config from tensor2tensor/utils/model_builder.py"""
import tensorflow as tf
from tensor2tensor.utils import yellowfin
import math
import numpy as np


class TransformerOptimizer:

    def __init__(self, loss, hparams, global_step, train_steps=500000, num_worker=1):
        learning_rate = hparams.learning_rate * learning_rate_decay(
            hparams, global_step, num_worker_replicas=num_worker, num_train_steps=train_steps)
        learning_rate /= math.sqrt(float(num_worker))

        opt = _ConditionalOptimizer(hparams.optimizer, learning_rate, hparams)
        self.train_op = tf.contrib.layers.optimize_loss(
            name="training",
            loss=loss,
            global_step=global_step,
            learning_rate=learning_rate,
            clip_gradients=hparams.clip_grad_norm or None,
            gradient_noise_scale=hparams.grad_noise_scale or None,
            optimizer=opt,
        )

    def get_op(self):
        return self.train_op


class _ConditionalOptimizer(tf.train.Optimizer):
  """Conditional optimizer."""

  def __init__(self, optimizer_name, lr, hparams):
    if optimizer_name == "Adam":
      # We change the default epsilon for Adam and re-scale lr.
      # Using LazyAdam as it's much faster for large vocabulary embeddings.
      self._opt = tf.contrib.opt.LazyAdamOptimizer(
          lr / 500.0,
          beta1=hparams.optimizer_adam_beta1,
          beta2=hparams.optimizer_adam_beta2,
          epsilon=hparams.optimizer_adam_epsilon)
    elif optimizer_name == "Momentum":
      self._opt = tf.train.MomentumOptimizer(
          lr, momentum=hparams.optimizer_momentum_momentum)
    elif optimizer_name == "YellowFin":
      tf.logging.info("Init YellowFin Optimizer.")
      self._opt = yellowfin.YellowFinOptimizer(
          learning_rate=lr, momentum=hparams.optimizer_momentum_momentum)
    else:
      self._opt = tf.contrib.layers.OPTIMIZER_CLS_NAMES[optimizer_name](lr)

  def compute_gradients(self, loss, var_list, colocate_gradients_with_ops):
    return self._opt.compute_gradients(
        loss, var_list, colocate_gradients_with_ops=colocate_gradients_with_ops)

  def apply_gradients(self, gradients, global_step=None, name=None):
    return self._opt.apply_gradients(
        gradients, global_step=global_step, name=name)


def learning_rate_decay(hparams, global_step, num_worker_replicas=1, num_train_steps=1):
  """Inverse-decay learning rate until warmup_steps, then decay."""
  warmup_steps = tf.to_float(
      hparams.learning_rate_warmup_steps * num_worker_replicas)
  step = tf.to_float(global_step)
  if hparams.learning_rate_decay_scheme == "noam":
    return 5000.0 * hparams.hidden_size**-0.5 * tf.minimum(
        (step + 1) * warmup_steps**-1.5, (step + 1)**-0.5)
  elif hparams.learning_rate_decay_scheme == "exp100k":
    return 0.94**(step // 100000)
  elif hparams.learning_rate_decay_scheme == "cosine":
    cycle_steps = hparams.learning_rate_cosine_cycle_steps
    return 0.5 * (1 + tf.cos(np.pi * (step % cycle_steps) / cycle_steps))
  elif hparams.learning_rate_decay_scheme == "cyclelinear10x":
    # Cycle the rate linearly by 10x every warmup_steps, up and down.
    cycle_steps = hparams.learning_rate_warmup_steps
    cycle_position = step % (2 * cycle_steps)
    cycle_position = tf.to_float(  # Normalize to the interval [-1, 1].
        cycle_position - cycle_steps) / float(cycle_steps)
    cycle_position = 1.0 - tf.abs(cycle_position)  # 0 to 1 and back to 0.
    return (cycle_position + 0.1) * 3.0  # 10x difference each cycle (0.3-3).

  inv_base = tf.exp(tf.log(0.01) / warmup_steps)
  inv_decay = inv_base**(warmup_steps - step)
  if hparams.learning_rate_decay_scheme == "sqrt":
    decay = _sqrt_decay(step - warmup_steps)
  elif hparams.learning_rate_decay_scheme == "exp10k":
    decay = _exp_decay_after(step - warmup_steps, 0.9995,
                             num_train_steps - warmup_steps - 10000)
  elif hparams.learning_rate_decay_scheme == "exp50k":
    decay = _exp_decay_after(step - warmup_steps, 0.99995,
                             num_train_steps - warmup_steps - 50000)
  elif hparams.learning_rate_decay_scheme == "exp500k":
    decay = _exp_decay_after(step - warmup_steps, 0.9999955,
                             num_train_steps - warmup_steps - 500000)
  elif hparams.learning_rate_decay_scheme == "none":
    decay = tf.constant(1.0)
  else:
    raise ValueError("Unrecognized learning rate decay scheme: %s" %
                     hparams.learning_rate_decay_scheme)
  return tf.cond(
      step < warmup_steps,
      lambda: inv_decay,
      lambda: decay,
      name="learning_rate_decay_warump_cond")


def _sqrt_decay(step):
  """Decay like 1 / sqrt(step), multiplied by 500 to normalize."""
  return 500.0 / tf.sqrt(tf.maximum(step, 1.0))


def _exp_decay_after(step, rate, from_which_step):
  """Decay exponentially by rate (per step) starting at from_which_step."""
  return tf.cond(
      step < from_which_step,
      lambda: tf.constant(1.0),
      lambda: rate**(step - from_which_step),
      name="exponential_decay_step_cond")