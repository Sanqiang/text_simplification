from data_generator.train_data import TrainData
from model.graph import Graph
from model.model_config import DefaultConfig, DefaultTrainConfig, WikiDressLargeTrainConfig
from data_generator.vocab import Vocab
from util import session
from util import constant

import tensorflow as tf
import math


def get_graph_train_data(data,
                   sentence_simple_input,
                   sentence_complex_input,
                   model_config):
    input_feed = {}
    voc = Vocab(model_config)

    tmp_sentence_simple, tmp_sentence_complex = [], []
    for i in range(model_config.batch_size):
        sentence_simple, sentence_complex = data.get_data_sample()

        # PAD zeros
        if len(sentence_simple) < model_config.max_simple_sentence:
            num_pad = model_config.max_simple_sentence - len(sentence_simple)
            sentence_simple.extend(num_pad * [voc.encode(constant.SYMBOL_PAD)])
        else:
            sentence_simple = sentence_simple[:model_config.max_simple_sentence]

        if len(sentence_complex) < model_config.max_complex_sentence:
            num_pad = model_config.max_complex_sentence - len(sentence_complex)
            sentence_complex.extend(num_pad * [voc.encode(constant.SYMBOL_PAD)])
        else:
            sentence_complex = sentence_complex[:model_config.max_complex_sentence]

        tmp_sentence_simple.append(sentence_simple)
        tmp_sentence_complex.append(sentence_complex)

    for step in range(model_config.max_simple_sentence):
        input_feed[sentence_simple_input[step].name] = [tmp_sentence_simple[batch_idx][step]
                                                        for batch_idx in range(model_config.batch_size)]
    for step in range(model_config.max_complex_sentence):
        input_feed[sentence_complex_input[step].name] = [tmp_sentence_complex[batch_idx][step]
                                                         for batch_idx in range(model_config.batch_size)]

    return input_feed


def train(model_config=None):
    model_config = (DefaultConfig()
                    if model_config is None else model_config)
    data = TrainData(model_config)
    graph = Graph(data, True, model_config)
    graph.create_model()

    sv = tf.train.Supervisor(logdir=model_config.logdir,
                             global_step=graph.global_step,
                             saver=graph.saver)
    sess = sv.PrepareSession(config=session.get_session_config(model_config))
    while True:
        input_feed = get_graph_train_data(data,
                                    graph.sentence_simple_input_placeholder,
                                    graph.sentence_complex_input_placeholder,
                                    model_config)

        fetches = [graph.train_op, graph.loss, graph.global_step, graph.decoder_target_list]
        _, loss, step, result = sess.run(fetches, input_feed)
        perplexity = math.exp(loss)
        print('Perplexity:\t%f at step %d.' % (perplexity, step))

        if step % model_config.model_save_freq == 0:
            graph.saver.save(sess, model_config.outdir + '/model.ckpt-%d' % step)
            # f = open(model_config.outdir + '/output_model.ckpt-%d' % step, 'w')
            # f.write(decode_output(result, data))
            # f.flush()


if __name__ == '__main__':
    train(WikiDressLargeTrainConfig())
