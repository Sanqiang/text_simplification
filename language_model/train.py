import sys
sys.path.insert(0,'/ihome/hdaqing/saz31/sanqiang/text_simplification')

import tensorflow as tf

from language_model.lm_data_generator import LM_Data
from language_model.transformer_lm import TransformerLM
from util import constant
from util.sys_moniter import print_cpu_memory, print_gpu_memory

from datetime import datetime

from language_model.lm_arguments import get_args

args = get_args()


def get_train_data(data, objs):
    input_feed = {}
    pad_id = [data.vocab.encode(constant.SYMBOL_PAD)]
    for obj in objs:
        tmp_sentence = []
        for i in range(args.batch_size):
            sentence = next(data.get_data_sample_it())
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

def train():
    data = LM_Data()
    graph = TransformerLM()
    graph.create_model_multigpu(data)
    sv = tf.train.Supervisor(logdir=args.logdir,
                             global_step=graph.global_step,
                             saver=graph.saver,
                             save_model_secs=600)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = sv.PrepareSession(config=config)
    perplexitys = []
    start_time = datetime.now()
    while True:
        input_feed = get_train_data(
            data,
            graph.objs)
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

        if step % 1000 == 0:
            print_gpu_memory()
            print_cpu_memory()

if __name__ == '__main__':
    train()