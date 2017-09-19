from data_generator.val_data import ValData
from model.graph import Graph
from model.model_config import DefaultConfig, DefaultTestConfig, WikiDressLargeTestConfig
from data_generator.vocab import Vocab
from util import constant
from util import session
from util.checkpoint import copy_ckpt_to_modeldir

from nltk.translate.bleu_score import sentence_bleu
import tensorflow as tf
import math
import numpy as np
import time


def get_graph_val_data(sentence_simple_input,
                       sentence_complex_input,
                       model_config, it):
    input_feed = {}
    voc = Vocab(model_config)

    tmp_sentence_simple, tmp_sentence_complex = [], []
    tmp_ref = [[] for _ in range(model_config.num_refs)]
    for i in range(model_config.batch_size):
        sentence_simple, sentence_complex, ref = next(it)
        if sentence_simple is None:
            # End of data set
            return None, None, None

        for i_ref in range(model_config.num_refs):
            tmp_ref[i_ref].append(ref[i_ref])

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
    val_data = ValData(model_config, model_config.vocab_simple, model_config.vocab_complex)
    graph = Graph(val_data, False, model_config)
    graph.create_model()

    while True:
        ibleus= []
        perplexitys = []
        decode_outputs = []
        it = val_data.get_data_iter()

        def init_fn(session):
            while True:
                try:
                    ckpt = copy_ckpt_to_modeldir(model_config)
                    graph.saver.restore(session, ckpt)
                    break
                except FileNotFoundError as exp:
                    print(str(exp) + '\nWait for 1 minutes.')
                    time.sleep(60)
            print('Restore ckpt:%s.' % ckpt)

        sv = tf.train.Supervisor(init_fn=init_fn)
        sess = sv.PrepareSession(config=session.get_session_config(model_config))
        while True:
            input_feed, sentence_simple, ref = get_graph_val_data(
                graph.sentence_simple_input_placeholder,
                graph.sentence_complex_input_placeholder,
                model_config, it)

            if input_feed is None:
                break

            fetches = [graph.decoder_target_list, graph.loss, graph.global_step]
            target, loss, step = sess.run(fetches, input_feed)
            batch_perplexity = math.exp(loss)
            perplexitys.append(batch_perplexity)

            target_output = decode_to_output(target, val_data)
            target = decode(target, val_data.vocab_simple)
            sentence_simple = decode(sentence_simple, val_data.vocab_simple)
            for ref_i in range(model_config.num_refs):
                ref[ref_i] = decode(ref[ref_i], val_data.vocab_simple)

            for batch_i in range(model_config.batch_size):
                # Compute iBLEU
                try:
                    batch_bleu_i = sentence_bleu(target[batch_i], sentence_simple[batch_i])
                    batch_bleu_rs = []
                    for ref_i in range(model_config.num_refs):
                        batch_bleu_rs.append(
                            sentence_bleu(target[batch_i], ref[ref_i][batch_i]))
                    if len(batch_bleu_rs) > 0:
                        batch_bleu_r = np.mean(batch_bleu_rs)
                        batch_ibleu = batch_bleu_r * 0.9 + batch_bleu_i * 0.1
                    else:
                        batch_ibleu = batch_bleu_i
                except Exception as e:
                    print('Bleu exception:\t' + str(e))
                    batch_ibleu = 0
                ibleus.append(batch_ibleu)
                # print('Batch iBLEU: \t%f.' % batch_ibleu)
            decode_outputs.append(target_output)

        ibleu = np.mean(ibleus)
        perplexity = np.mean(perplexitys)
        print('Current iBLEU: \t%f' % ibleu)
        print('Current perplexity: \t%f' % perplexity)
        print('Current eval done!')
        f = open(model_config.modeldir + '/step' + str(step) + 'ibleu' + str(ibleu), 'w')
        f.write(str(ibleu))
        f.write('\t')
        f.write(str(perplexity))
        f.flush()
        f = open(model_config.modeldir + '/step' + str(step) + '-ibleu' + str(ibleu) + '.result', 'w')
        f.write('\n'.join(decode_outputs))
        f.flush()


def decode_to_output(target, data):
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
        if constant.SYMBOL_END in decode_result:
            eos = decode_result.index(constant.SYMBOL_END)
            decode_result = decode_result[:eos]
        if len(decode_result) > 0 and decode_result[0] == constant.SYMBOL_START:
            decode_result = decode_result[1:]
        decode_results.append(decode_result)
    return decode_results


if __name__ == '__main__':
    eval(WikiDressLargeTestConfig())
