import sys
sys.path.insert(0,'/ihome/hdaqing/saz31/sanqiang/text_simplification')

import tensorflow as tf

from language_model.lm_data_generator import LM_Data
from language_model.transformer_lm import TransformerLM
from util import constant
from util.sys_moniter import print_cpu_memory, print_gpu_memory
from os.path import exists
from os import makedirs
from datetime import datetime
from language_model.lm_arguments import get_args
from model.eval import copy_ckpt_to_modeldir
import numpy as np

args = get_args()


def eval():
    def get_eval_data(data, objs, it):
        input_feed = {}
        if args.subword_vocab_size > 0:
            pad_id = data.vocab.encode(constant.SYMBOL_PAD)
        else:
            pad_id = [data.vocab.encode(constant.SYMBOL_PAD)]
        is_end = False
        for obj in objs:
            tmp_sentence = []
            for i in range(args.batch_size):
                sentence = next(it)
                if sentence is None:
                    is_end = True
                else:
                    if len(sentence) < args.max_sent_len:
                        num_pad = args.max_sent_len - len(sentence)
                        sentence.extend(num_pad * pad_id)
                    else:
                        sentence = sentence[:args.max_sent_len]
                    tmp_sentence.append(sentence)

            for step in range(args.max_sent_len):
                input_feed[obj['sentence_inputs'][step].name] = [tmp_sentence[batch_idx][step]
                                                                 for batch_idx in range(len(tmp_sentence))]
        return input_feed, is_end

    if not exists(args.resultdir):
        makedirs(args.resultdir)

    try:
        ckpt = copy_ckpt_to_modeldir(args.modeldir, args.logdir)
    except FileNotFoundError:
        print('No current ckpt for eval.')
        return

    tf.reset_default_graph()
    data = LM_Data()
    graph = TransformerLM()
    graph.create_model_multigpu(data, False)
    def init_fn(session):
        graph.saver.restore(session, ckpt)
        print('Restore ckpt:%s.' % ckpt)
    sv = tf.train.Supervisor(init_fn=init_fn)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = sv.PrepareSession(config=config)
    perplexitys = []
    it = data.get_evaldata_sample_it()
    while True:
        input_feed, is_end = get_eval_data(
            data,
            graph.objs, it)
        if is_end:
            break
        fetches = [graph.loss, graph.global_step,
                   graph.perplexity]
        loss, step, perplexity = sess.run(fetches, input_feed)
        perplexitys.append(perplexity)

    val = 'step' + str(step) + 'perplexity' + str(np.mean(perplexitys))
    f = open(args.resultdir + val, 'w')
    f.write(val)
    f.close()


def train():
    def get_train_data(data, objs, it):
        input_feed = {}
        if args.subword_vocab_size > 0:
            pad_id = data.vocab.encode(constant.SYMBOL_PAD)
        else:
            pad_id = [data.vocab.encode(constant.SYMBOL_PAD)]
        for obj in objs:
            tmp_sentence = []
            for i in range(args.batch_size):
                sentence = next(it)
                if len(sentence) < args.max_sent_len:
                    num_pad = args.max_sent_len - len(sentence)
                    sentence.extend(num_pad * pad_id)
                else:
                    sentence = sentence[:args.max_sent_len]
                tmp_sentence.append(sentence)

            for step in range(args.max_sent_len):
                input_feed[obj['sentence_inputs'][step].name] = [tmp_sentence[batch_idx][step]
                                                                 for batch_idx in range(args.batch_size)]
        return input_feed

    data = LM_Data()
    graph = TransformerLM()
    graph.create_model_multigpu(data, True)
    sv = tf.train.Supervisor(logdir=args.logdir,
                             global_step=graph.global_step,
                             saver=graph.saver,
                             save_model_secs=600)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = sv.PrepareSession(config=config)
    perplexitys = []
    start_time = datetime.now()
    it = data.get_data_sample_it()
    while True:
        input_feed = get_train_data(
            data,
            graph.objs, it)
        fetches = [graph.train_op, graph.loss, graph.global_step,
                   graph.perplexity]
        _, loss, step, perplexity = sess.run(fetches, input_feed)
        perplexitys.append(perplexity)

        if step % 100 == 0:
            end_time = datetime.now()
            time_span = end_time - start_time
            start_time = end_time
            print('Perplexity:\t%f at step %d using %s.' % (perplexity, step, time_span))
            perplexitys.clear()

        if step % 10000 == 0:
            print_gpu_memory()
            print_cpu_memory()
            eval()


if __name__ == '__main__':
    train()