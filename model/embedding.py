from model.model_config import DefaultConfig

import tensorflow as tf
import math

class Embedding:
    def __init__(self, voc, model_config):
        self.model_config = model_config
        self.voc = voc

    def get_embedding(self):
        sqrt3 = math.sqrt(3)
        emb = tf.Variable(tf.random_uniform(
            [len(self.voc.i2w), self.model_config.dimension],
            -sqrt3, sqrt3), name='embedding')
        return emb

