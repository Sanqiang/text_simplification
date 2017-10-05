# For fix slurm cannot load PYTHONPATH
import sys
sys.path.insert(0,'/ihome/hdaqing/saz31/sanqiang/text_simplification')


from data_generator.train_data import TrainData
from model.transformer import TransformerGraph
from model.seq2seq import Seq2SeqGraph
from model.model_config import DefaultConfig, DefaultTrainConfig, WikiDressLargeTrainConfig, list_config
from data_generator.vocab import Vocab
from util import session
from util import constant
from model.eval import decode_to_output, decode

import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
from util.arguments import get_args


args = get_args()


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

    return input_feed, tmp_sentence_simple, tmp_sentence_complex


def train(model_config=None):
    model_config = (DefaultConfig()
                    if model_config is None else model_config)
    data = TrainData(model_config)

    graph = None
    if model_config.framework == 'transformer':
        graph = TransformerGraph(data, True, model_config)
    elif model_config.framework == 'seq2seq':
        graph = Seq2SeqGraph(data, True, model_config)
    else:
        raise NotImplementedError('Unknown Framework.')

    graph.create_model()

    if model_config.warm_start:
        partial_restore_ckpt = slim.assign_from_checkpoint_fn(
            model_config.warm_start, slim.get_variables_to_restore(),
            ignore_missing_vars=True, reshape_variables=False)

    def init_fn(session):
        if model_config.pretrained_embedding is not None:
            input_feed = {graph.embed_simple_placeholder: data.pretrained_emb_simple,
                          graph.embed_complex_placeholder: data.pretrained_emb_complex}
            session.run([graph.replace_emb_complex, graph.replace_emb_simple], input_feed)
            print('Replace Pretrained Word Embedding.')

        if model_config.warm_start:
            try:
                graph.saver.restore(session, model_config.warm_start)
            except Exception as ex:
                print('Fully restore failed, use partial restore instead. \n %s' % str(ex))
                partial_restore_ckpt(session)

            print('Warm start with checkpoint %s' % model_config.warm_start)

    sv = tf.train.Supervisor(logdir=model_config.logdir,
                             global_step=graph.global_step,
                             saver=graph.saver,
                             init_fn=init_fn,
                             save_model_secs=model_config.save_model_secs)
    sess = sv.PrepareSession(config=session.get_session_config(model_config))
    while True:
        input_feed, sentence_simple, sentence_complex = get_graph_train_data(
            data,
            graph.sentence_simple_input_placeholder,
            graph.sentence_complex_input_placeholder,
            model_config)

        fetches = [graph.train_op, graph.loss, graph.global_step, graph.decoder_target_list]
        _, loss, step, result = sess.run(fetches, input_feed)
        perplexity = math.exp(loss)
        print('Perplexity:\t%f at step %d.' % (perplexity, step))

        if step % model_config.model_save_freq == 0:
            graph.saver.save(sess, model_config.outdir + '/model.ckpt-%d' % step)
            f = open(model_config.outdir + '/output_model.ckpt-%d' % step, 'w', encoding='utf-8')
            result = decode(result, data.vocab_simple)
            sentence_simple = decode(sentence_simple, data.vocab_simple)
            sentence_complex = decode(sentence_complex, data.vocab_complex)
            r = decode_to_output(result, sentence_simple, sentence_complex, model_config.batch_size)
            f.write(r)
            f.flush()


if __name__ == '__main__':
    config = None
    if args.mode == 'dummy':
        config = DefaultTrainConfig()
    elif args.mode == 'dress':
        config = WikiDressLargeTrainConfig()
    print(list_config(config))
    train(config)
