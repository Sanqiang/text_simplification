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
from util.checkpoint import find_train_ckptpaths, backup_log
from model.eval import eval, get_ckpt

import tensorflow as tf
import tensorflow.contrib.slim as slim
from util.arguments import get_args


args = get_args()


def get_graph_train_data(
        data,
        sentence_simple_input,
        sentence_complex_input,
        sentence_simple_input_weight,
        model_config):
    input_feed = {}
    # Reserved section of vocabuary are same.
    voc = data.vocab_simple

    if model_config.subword_vocab_size > 0:
        pad_id = voc.encode(constant.SYMBOL_PAD)
    else:
        pad_id = [voc.encode(constant.SYMBOL_PAD)]

    tmp_sentence_simple, tmp_sentence_complex, tmp_sentence_simple_weight = [], [], []
    for i in range(model_config.batch_size):
        sentence_simple, sentence_complex, sentence_simple_weight = data.get_data_sample()

        # PAD zeros
        if len(sentence_simple) < model_config.max_simple_sentence:
            num_pad = model_config.max_simple_sentence - len(sentence_simple)
            sentence_simple.extend(num_pad * pad_id)
        else:
            sentence_simple = sentence_simple[:model_config.max_simple_sentence]

        if len(sentence_complex) < model_config.max_complex_sentence:
            num_pad = model_config.max_complex_sentence - len(sentence_complex)
            sentence_complex.extend(num_pad * pad_id)
        else:
            sentence_complex = sentence_complex[:model_config.max_complex_sentence]

        tmp_sentence_simple.append(sentence_simple)
        tmp_sentence_complex.append(sentence_complex)

        if sentence_simple_weight is not None:
            if len(sentence_simple_weight) < model_config.max_simple_sentence:
                num_pad = model_config.max_simple_sentence - len(sentence_simple_weight)
                sentence_simple_weight.extend(num_pad * [voc.encode(constant.SYMBOL_PAD)])
            else:
                sentence_simple_weight = sentence_simple[:model_config.max_simple_sentence]
            tmp_sentence_simple_weight.append(sentence_simple_weight)

    for step in range(model_config.max_simple_sentence):
        input_feed[sentence_simple_input[step].name] = [tmp_sentence_simple[batch_idx][step]
                                                        for batch_idx in range(model_config.batch_size)]
    for step in range(model_config.max_complex_sentence):
        input_feed[sentence_complex_input[step].name] = [tmp_sentence_complex[batch_idx][step]
                                                         for batch_idx in range(model_config.batch_size)]

    if tmp_sentence_simple_weight:
        for step in range(model_config.max_simple_sentence):
            input_feed[sentence_simple_input_weight[step].name] = [tmp_sentence_simple_weight[batch_idx][step]
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

    if model_config.change_optimizer:
        # Back up Log
        backup_log(model_config.logdir)
        ckpt_model = find_train_ckptpaths(model_config.outdir)

    ckpt_path = None
    if model_config.warm_start:
        ckpt_path = model_config.warm_start
        var_list = slim.get_variables_to_restore()
    elif model_config.change_optimizer:
        ckpt_path = ckpt_model
        var_list = [v for v in slim.get_variables_to_restore() if 'optim' not in v.name]
    if ckpt_path is not None:
        partial_restore_ckpt = slim.assign_from_checkpoint_fn(
            ckpt_path, var_list,
            ignore_missing_vars=True, reshape_variables=False)

    def init_fn(session):
        if model_config.pretrained_embedding is not None and model_config.subword_vocab_size <= 0:
            input_feed = {graph.embed_simple_placeholder: data.pretrained_emb_simple,
                          graph.embed_complex_placeholder: data.pretrained_emb_complex}
            session.run([graph.replace_emb_complex, graph.replace_emb_simple], input_feed)
            print('Replace Pretrained Word Embedding.')

            del data.pretrained_emb_simple
            del data.pretrained_emb_complex

        # Restore ckpt either from warm start or automatically get when changing optimizer
        ckpt_path = None
        if model_config.warm_start:
            ckpt_path = model_config.warm_start
        if model_config.change_optimizer:
            ckpt_path = ckpt_model

        if ckpt_path is not None:
            if model_config.use_partial_restore:
                partial_restore_ckpt(session)
            else:
                try:
                    graph.saver.restore(session, ckpt_path)
                except Exception as ex:
                    print('Fully restore failed, use partial restore instead. \n %s' % str(ex))
                    partial_restore_ckpt(session)

            print('Warm start with checkpoint %s' % ckpt_path)

    sv = tf.train.Supervisor(logdir=model_config.logdir,
                             global_step=graph.global_step,
                             saver=graph.saver,
                             init_fn=init_fn,
                             save_model_secs=model_config.save_model_secs)
    sess = sv.PrepareSession(config=session.get_session_config(model_config))
    perplexitys = []
    while True:
        input_feed, sentence_simple, sentence_complex = get_graph_train_data(
            data,
            graph.sentence_simple_input_placeholder,
            graph.sentence_complex_input_placeholder,
            graph.sentence_simple_input_prior_placeholder,
            model_config)

        fetches = [graph.train_op, graph.loss, graph.global_step, graph.decoder_target_list,
                   graph.perplexity, graph.learning_rate]
        _, loss, step, result, perplexity, lr = sess.run(fetches, input_feed)
        perplexitys.append(perplexity)

        if step % model_config.model_print_freq == 0:
            print('Perplexity:\t%f at step %d with lr %s' % (perplexity, step, lr))
            perplexitys.clear()

        if step % model_config.model_eval_freq == 0:
            from model.model_config import SubValWikiEightRefConfig, SubTestWikiEightRefConfig
            from model.model_config import DefaultTestConfig, DefaultTestConfig2
            ckpt = get_ckpt(model_config.modeldir, model_config.logdir)
            if ckpt:
                if args.mode == 'dummy':
                    eval(DefaultTestConfig(), ckpt)
                    eval(DefaultTestConfig2(), ckpt)
                elif args.mode == 'dress' or args.mode == 'all':
                    # eval(SubValWikiEightRefConfig(), ckpt)
                    eval(SubTestWikiEightRefConfig(), ckpt)


if __name__ == '__main__':
    config = None
    if args.mode == 'dummy':
        config = DefaultTrainConfig()
    elif args.mode == 'dress':
        config = WikiDressLargeTrainConfig()
    print(list_config(config))
    train(config)
