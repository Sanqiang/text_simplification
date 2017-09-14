from data_generator.val_data import ValData
from model.graph import Graph
from model.model_config import DefaultConfig, DefaultTestConfig
from data_generator.vocab import Vocab
from util import constant
from util.checkpoint import copy_ckpt_to_modeldir

from nltk.translate.bleu_score import sentence_bleu
import tensorflow as tf
import math
import numpy as np


def get_graph_val_data(data,
                       sentence_simple_input,
                       sentence_complex_input,
                       model_config):
    input_feed = {}
    voc = Vocab()

    tmp_sentence_simple, tmp_sentence_complex = [], []
    tmp_ref = [[] for _ in range(model_config.num_refs)]
    for i in range(model_config.batch_size):
        sentence_simple, sentence_complex, ref = next(data.get_data_iter())
        if sentence_simple is None:
            # End of data set
            return None

        for i_ref in range(model_config.num_refs):
            tmp_ref[i_ref].append(ref[i])

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

    return input_feed, tmp_sentence_simple, tmp_ref


def eval(model_config=None):
    model_config = (DefaultConfig()
                    if model_config is None else model_config)
    val_data = ValData(model_config.vocab_simple, model_config.vocab_complex, model_config)
    graph = Graph(val_data, False, model_config)
    graph.create_model()

    while True:
        ibleu = []
        perplexity = []

        def init_supervisor(sess):
            ckpt = copy_ckpt_to_modeldir(model_config)
            graph.saver.restore(sess, ckpt)
            print('Restore ckpt:%s.' % ckpt)

        sv = tf.train.Supervisor(logdir=model_config.logdir,
                                 init_op=init_supervisor)
        sess = sv.PrepareSession()
        while True:
            input_feed, sentence_simple, ref = get_graph_val_data(val_data,
                                                                  graph.sentence_simple_input_placeholder,
                                                                  graph.sentence_complex_input_placeholder,
                                                                  model_config)
            if input_feed is None:
                break

            fetches = [graph.decoder_target_list, graph.loss, graph.global_step]
            target, loss, step = sess.run(fetches, input_feed)
            batch_perplexity = math.exp(loss)
            perplexity.append(batch_perplexity)

            target = decode(target, val_data.vocab_simple)
            sentence_simple = decode(sentence_simple, val_data.vocab_simple)
            for ref_i in range(model_config.num_refs):
                ref[ref_i] = decode(ref[ref_i], val_data.vocab_simple)

            # Print decode result
            # print(decode_output(target, data))

            for batch_i in range(model_config.batch_size):
                # Compute iBLEU
                batch_bleu_rs = []
                for ref_i in range(model_config.num_refs):
                    batch_bleu_rs.append(
                        sentence_bleu(target[batch_i], ref[ref_i][batch_i]))
                batch_bleu_r = np.mean(batch_bleu_rs)
                batch_bleu_i = sentence_bleu(target[batch_i], sentence_simple[batch_i])
                batch_ibleu = batch_bleu_r * 0.9 + batch_bleu_i * 0.1
                ibleu.append(batch_ibleu)

        print('Current iBLEU: \t%f' % np.mean(ibleu))
        print('Current perplexity: \t%f' % np.mean(perplexity))
        print('Current eval done!')


def decode_output(target, data):
    output = ''
    decode_results = decode(target, data.vocab_simple)
    for decode_result in decode_results:
        output = '\n'.join([output, ' '.join(decode_result)])
    return output


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
