from data_generator.data import Data
from model.graph import Graph, get_graph_data
from model.model_config import DefaultConfig

import tensorflow as tf
import math

def train(model_config=None):
    model_config = (DefaultConfig()
                    if model_config is None else model_config)
    data = Data(model_config.dataset_simple, model_config.dataset_complex,
                model_config.vocab_simple, model_config.vocab_complex)
    graph = Graph(data, model_config)
    graph.create_model()

    sv = tf.train.Supervisor(logdir=model_config.logdir,
                             global_step=graph.global_step,
                             saver=graph.saver)
    sess = sv.PrepareSession(master='')
    while True:
        input_feed = get_graph_data(data,
                                    graph.sentence_simple_input_placeholder,
                                    graph.sentence_complex_input_placeholder,
                                    model_config)

        fetches = [graph.train_op, graph.loss, graph.global_step]
        _, loss, step = sess.run(fetches, input_feed)
        perplexity = math.exp(loss)
        print('Perplexity:\t%f at step %d.' % (perplexity, step))

        step += 1
        if step % 100 == 0:
            graph.saver.save(sess, model_config.outdir + '/model.ckpt-%d' % step)


if __name__ == '__main__':
    train()