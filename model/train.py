# For fix slurm cannot load PYTHONPATH
import sys
# sys.path.insert(0,'/ihome/hdaqing/saz31/sanqiang/text_simplification')
# sys.path.insert(0,'/home/hed/text_simp_proj/text_simplification')
# sys.path.insert(0,'/ihome/hdaqing/saz31/ts/text_simplification')
sys.path.insert(0,'/ihome/hdaqing/saz31/ts_0924/text_simplification')


from data_generator.train_data import TrainData, TfExampleTrainDataset
from model.transformer import TransformerGraph
from model.seq2seq import Seq2SeqGraph
from model.model_config import DefaultConfig, DefaultTrainConfig, list_config
from model.model_config import WikiDressLargeNewTrainDefault,WikiDressLargeTrainDefault
from model.model_config import WikiTransTrainConfig, WikiTransBertTrainConfig, WikiTransDressTokenTrainConfig, WikiSplitBertTrainConfig, SentCompressBertTrainConfig, WikiCombineBertTrainConfig, WikiOriBertTrainConfig
from model.model_config import NewselaBertTrainConfig
from data_generator.vocab import Vocab
from util import session
from util import constant
from util import checkpoint
from model.eval import eval, get_ckpt, get_best_sari
from model.model_config import get_path
from model.bert import utils

import tensorflow as tf
import tensorflow.contrib.slim as slim
from util.arguments import get_args
from datetime import datetime
from util.sys_moniter import print_cpu_memory, print_gpu_memory, print_cpu_usage
from os.path import exists, dirname, join
from os import listdir, remove


args = get_args()


def get_graph_train_data(
        data,
        objs,
        model_config):
    input_feed = {}
    for obj in objs:
        (tmp_sentence_simple, tmp_sentence_complex,
         tmp_sentence_simple_weight, tmp_attn_weight,
         tmp_idxs, tmp_sups, tmp_sentence_simple_raw, tmp_sentence_complex_raw) = [], [], [], [], [], {}, [], []

        for i in range(model_config.batch_size):
            if not model_config.it_train:
                idx, obj_data, sup = data.get_data_sample()
            else:
                idx, obj_data, sup = next(data.data_it)

            tmp_sentence_simple.append(obj_data['words_simp'])
            tmp_sentence_complex.append(obj_data['words_comp'])
            tmp_idxs.append(idx)

            if 'rule' in model_config.memory or 'direct' in model_config.memory:
                if 'rule_id_input_placeholder' not in tmp_sups:
                    tmp_sups['rule_id_input_placeholder'] = []
                if 'rule_target_input_placeholder' not in tmp_sups:
                    tmp_sups['rule_target_input_placeholder'] = []

                cur_rule_id_input_placeholder = []
                cur_rule_target_input_placeholder = []
                for rule_tuple in sup['rules_target']:
                    rule_id = rule_tuple[0]
                    rule_targets = rule_tuple[1]
                    for target in rule_targets:
                        cur_rule_id_input_placeholder.append(rule_id)
                        cur_rule_target_input_placeholder.append(target)

                if len(cur_rule_id_input_placeholder) < model_config.max_cand_rules:
                    num_pad = model_config.max_cand_rules - len(cur_rule_id_input_placeholder)
                    cur_rule_id_input_placeholder.extend(num_pad * [0])
                    cur_rule_target_input_placeholder.extend(num_pad * [constant.SYMBOL_PAD])
                else:
                    cur_rule_id_input_placeholder = cur_rule_id_input_placeholder[:model_config.max_cand_rules]
                    cur_rule_target_input_placeholder = cur_rule_target_input_placeholder[:model_config.max_cand_rules]

                tmp_sups['rule_id_input_placeholder'].append(cur_rule_id_input_placeholder)
                tmp_sups['rule_target_input_placeholder'].append(cur_rule_target_input_placeholder)

        for step in range(model_config.max_simple_sentence):
            input_feed[obj['sentence_simple_input_placeholder'][step].name] = [tmp_sentence_simple[batch_idx][step]
                                                            for batch_idx in range(model_config.batch_size)]
        for step in range(model_config.max_complex_sentence):
            input_feed[obj['sentence_complex_input_placeholder'][step].name] = [tmp_sentence_complex[batch_idx][step]
                                                             for batch_idx in range(model_config.batch_size)]
        input_feed[obj['sentence_idxs'].name] = [tmp_idxs[batch_idx] for batch_idx in range(model_config.batch_size)]

        if 'rule' in model_config.memory or 'direct' in model_config.memory:
            for step in range(model_config.max_cand_rules):
                input_feed[obj['rule_id_input_placeholder'][step].name] = [
                    tmp_sups['rule_id_input_placeholder'][batch_idx][step]
                    for batch_idx in range(model_config.batch_size)]
                input_feed[obj['rule_target_input_placeholder'][step].name] = [
                    tmp_sups['rule_target_input_placeholder'][batch_idx][step]
                    for batch_idx in range(model_config.batch_size)]

    return input_feed


def train(model_config=None):
    model_config = (DefaultConfig()
                    if model_config is None else model_config)

    if model_config.fetch_mode == 'tf_example_dataset':
        data = TfExampleTrainDataset(model_config)
    else:
        data = TrainData(model_config)

    if model_config.framework == 'transformer':
        graph = TransformerGraph(data, True, model_config)
    elif model_config.framework == 'seq2seq':
        graph = Seq2SeqGraph(data, True, model_config)
    else:
        raise NotImplementedError('Unknown Framework.')
    graph.create_model_multigpu()

    ckpt_path = None
    if model_config.warm_start:
        ckpt_path = model_config.warm_start
        var_list = slim.get_variables_to_restore()
    if ckpt_path is not None:
        # Handling missing vars by ourselves
        available_vars = {}
        reader = tf.train.NewCheckpointReader(ckpt_path)
        var_dict = {var.op.name: var for var in var_list}
        for var in var_dict:
            if 'global_step' in var and 'optim' not in model_config.warm_config:
                print('Ignore var:', var)
                continue
            if 'optimization' in var and 'optim' not in model_config.warm_config:
                print('Ignore var:', var)
                continue
            if reader.has_tensor(var):
                var_ckpt = reader.get_tensor(var)
                var_cur = var_dict[var]
                if any([var_cur.shape[i] != var_ckpt.shape[i] for i in range(len(var_ckpt.shape))]):
                    print('Variable missing due to shape.', var)
                else:
                    available_vars[var] = var_dict[var]
            else:
                print('Variable missing:', var)

        partial_restore_ckpt = slim.assign_from_checkpoint_fn(
            ckpt_path, available_vars,
            ignore_missing_vars=False, reshape_variables=False)

    if model_config.bert_mode:
        bert_restore_ckpt = utils.restore_bert(ckpt=model_config.bert_ckpt)

    if 'direct' in model_config.memory:
        bert_direct_restore_ckpt = utils.restore_bert(ckpt=model_config.bert_ckpt, model='direct/')

    sess = tf.train.MonitoredTrainingSession(
            checkpoint_dir=model_config.logdir,
            save_checkpoint_secs=model_config.save_model_secs,
            config=session.get_session_config(model_config),
            hooks=[tf.train.CheckpointSaverHook(
                model_config.logdir,
                save_secs=model_config.save_model_secs,
                saver=graph.saver)],
            save_summaries_steps=None,
            save_summaries_secs=None,  # Disable tf.summary
    )

    if checkpoint.is_fresh_run(model_config.logdir) and 'init' in model_config.bert_mode:
        if model_config.bert_mode:
            if 'direct' in model_config.memory:
                bert_direct_restore_ckpt(sess)
            # else:
            bert_restore_ckpt(sess)
            print('BERT init')

    if checkpoint.is_fresh_run(model_config.logdir):
        if ckpt_path is not None:
            partial_restore_ckpt(sess)
            print('Restore from %s' % ckpt_path)

    perplexitys = []
    start_time = datetime.now()

    # Intialize tf example dataset reader
    if model_config.fetch_mode == 'tf_example_dataset':
        if model_config.dmode == 'listalter':
            assert type(data.training_init_op) == list
            for init_op in data.training_init_op:
                sess.run(init_op)
        else:
            sess.run(data.training_init_op)
            print('Init dataset interator.')
            if model_config.dmode == 'alter':
                sess.run(data.training_init_op2)
                print('Init dataset2 interator.')

    # with tf.contrib.tfprof.ProfileContext('/zfs1/hdaqing/saz31/text_simplification_0924/bertbaseal2_ls/profile') as pctx:
    while True:
        fetches = [graph.train_op, graph.loss, graph.global_step,
                   graph.perplexity, graph.ops, graph.increment_global_step, graph.loss_style]
        if model_config.fetch_mode:
            _, loss, step, perplexity, _, _, loss_style = sess.run(fetches)
        else:
            input_feed = get_graph_train_data(
                data,
                graph.objs,
                model_config)
            _, loss, step, perplexity, _, _ = sess.run(fetches, input_feed)
        perplexitys.append(perplexity)

        if step % model_config.model_print_freq == 0:
            end_time = datetime.now()
            time_span = end_time - start_time
            start_time = end_time
            print('Perplexity:\t%f at step %d using %s.' % (perplexity, step, time_span))
            if 'pred' in model_config.tune_mode:
                print('Loss:%s\tLoss_tyle:%s' % (loss, loss_style))
            perplexitys.clear()
            if step / model_config.model_print_freq == 1:
                print_cpu_usage()
                print_cpu_memory()
                print_gpu_memory()

        #if step % (100 * model_config.model_print_freq) == 0:
        #    graph.saver.save(sess, join(model_config.logdir, 'bk.ckpt-', step))

        if model_config.model_eval_freq > 0 and step % model_config.model_eval_freq == 0:
            if args.mode == 'dress':
                from model.model_config import WikiDressLargeDefault, WikiDressLargeEvalDefault, \
                    WikiDressLargeTestDefault
                model_config = WikiDressLargeDefault()
                ckpt = get_ckpt(model_config.modeldir, model_config.logdir)

                vconfig = WikiDressLargeEvalDefault()
                best_sari = get_best_sari(vconfig.resultdir)
                sari_point = eval(vconfig, ckpt)
                eval(WikiDressLargeTestDefault(), ckpt)
                if args.memory is not None and 'rule' in args.memory:
                    for rcand in [15, 30, 50]:
                        vconfig.max_cand_rules = rcand
                        vconfig.resultdir = get_path(
                            '../' + vconfig.output_folder + '/result/eightref_val_cand' + str(rcand),
                            vconfig.environment)
                        eval(vconfig, ckpt)
                print('=====================Current Best SARI:%s=====================' % best_sari)
                if float(sari_point) < best_sari:
                    remove(ckpt + '.index')
                    remove(ckpt + '.meta')
                    remove(ckpt + '.data-00000-of-00001')
                    print('remove ckpt:%s' % ckpt)
                else:
                    for file in listdir(model_config.modeldir):
                        step = ckpt[ckpt.rindex('model.ckpt-') + len('model.ckpt-'):-1]
                        if step not in file:
                            remove(model_config.modeldir + file)
                    print('Get Best Model, remove ckpt except:%s.' % ckpt)

if __name__ == '__main__':
    config = None
    if args.mode == 'dummy':
        config = DefaultTrainConfig()
    elif args.mode == 'dressnew':
        config = WikiDressLargeNewTrainDefault()
    elif args.mode == 'dress':
        config = WikiDressLargeTrainDefault()
    elif args.mode == 'trans':
        config = WikiTransTrainConfig()
    elif args.mode == 'transbert':
        config = WikiTransBertTrainConfig()
    elif args.mode == 'dresstk':
        config = WikiTransDressTokenTrainConfig()
    elif args.mode == 'wikisplit':
        config = WikiSplitBertTrainConfig()
    elif args.mode == 'newsela':
        config = NewselaBertTrainConfig()
    elif args.mode == 'sentcompress':
        config = SentCompressBertTrainConfig()
    elif args.mode == 'comb':
        config = WikiCombineBertTrainConfig()
    elif args.mode == 'transbert_ori':
        config = WikiOriBertTrainConfig()
    else:
        raise ValueError('mode unknown')
    print(list_config(config))
    train(config)
