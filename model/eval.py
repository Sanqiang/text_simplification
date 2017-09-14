from data_generator.data import Data
from model.graph import Graph, get_graph_data
from model.model_config import DefaultConfig, DefaultTestConfig
from util import constant

import tensorflow as tf
import math

def eval(model_config=None):
    model_config = (DefaultConfig()
                    if model_config is None else model_config)
    data = Data(model_config.dataset_simple, model_config.dataset_complex,
                model_config.vocab_simple, model_config.vocab_complex)
    graph = Graph(data, False, model_config)
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

        fetches = [graph.decoder_target_list, graph.loss, graph.global_step]
        target, loss, step = sess.run(fetches, input_feed)
        perplexity = math.exp(loss)
        print('Perplexity:\t%f at step %d.' % (perplexity, step))

        #Decode
        decode_results = decode(target, data.vocab_simple)
        for decode_result in decode_results:
            print(' '.join(decode_result))
        print('====================================')

def decode(target, voc):
    target = list(target)
    batch_size = len(target)
    decode_results = []
    for i in range(batch_size):
        decode_result = list(map(voc.describe, target[i]))
        if constant.SYMBOL_PAD in decode_result:
            eos = decode_result.index(constant.SYMBOL_PAD)
            decode_result = decode_result[:eos]
        decode_results.append(decode_result)
    return decode_results


if __name__ == '__main__':
    eval(DefaultTestConfig())