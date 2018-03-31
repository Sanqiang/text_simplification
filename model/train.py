# For fix slurm cannot load PYTHONPATH
import sys
# sys.path.insert(0,'/ihome/hdaqing/saz31/sanqiang/text_simplification')
# sys.path.insert(0,'/home/hed/text_simp_proj/text_simplification')
sys.path.insert(0,'/ihome/cs2770_s2018/maz54/ts/text_simplification')


from data_generator.train_data import TrainData
from model.transformer import TransformerGraph
from model.model_config import DefaultConfig, DefaultTrainConfig, WikiDressLargeTrainConfig, WikiDressSmallTrainConfig, list_config
from model.model_config import WikiTransTrainCfg, WikiTransValCfg, WikiTransLegacyTestCfg, WikiTransLegacyTrainCfg
from model.model_config import WikiDressLargeNewTrainDefault
from data_generator.vocab import Vocab
from util import session
from util import constant
from util.checkpoint import find_train_ckptpaths, backup_log
from model.eval import eval, get_ckpt

import tensorflow as tf
import tensorflow.contrib.slim as slim
from util.arguments import get_args
from datetime import datetime
from util.sys_moniter import print_cpu_memory, print_gpu_memory
from copy import deepcopy


args = get_args()


def get_graph_train_data(
        data,
        objs,
        model_config):
    input_feed = {}
    # Reserved section of vocabuary are same.

    for obj in objs:

        (tmp_sentence_simple, tmp_sentence_complex,
         tmp_sentence_simple_weight, tmp_attn_weight,
         tmp_idxs, tmp_sups, tmp_sentence_simple_raw, tmp_sentence_complex_raw) = [], [], [], [], [], {}, [], []

        if model_config.pointer_mode == 'ptr':
            batch_oov = {}
            batch_oov['w2i'] = {}
            batch_oov['i2w'] = []

        for i in range(model_config.batch_size):
            if not model_config.it_train:
                idx, obj_data = data.get_data_sample()
            else:
                raise NotImplemented
                idx, sentence_simple, sentence_complex, sentence_simple_weight, attn_weight, sup = next(data.data_it)

            tmp_sentence_simple.append(obj_data['words_simp'])
            tmp_sentence_complex.append(obj_data['words_comp'])
            tmp_idxs.append(idx)

            if model_config.pointer_mode == 'ptr':
                tmp_sentence_simple_raw.append(obj_data['words_raw_simp'])
                tmp_sentence_complex_raw.append(obj_data['words_raw_comp'])

            if model_config.memory == 'rule':
                if 'rule_id_input_placeholder' not in tmp_sups:
                    tmp_sups['rule_id_input_placeholder'] = []
                if 'rule_target_input_placeholder' not in tmp_sups:
                    tmp_sups['rule_target_input_placeholder'] = []

                cur_rule_id_input_placeholder = []
                cur_rule_target_input_placeholder = []
                for rule_tuple in sup['mem']:
                    rule_id = rule_tuple[0]
                    rule_targets = rule_tuple[1]
                    for target in rule_targets:
                        cur_rule_id_input_placeholder.append(rule_id)
                        cur_rule_target_input_placeholder.append(target)

                if len(cur_rule_id_input_placeholder) < model_config.max_cand_rules:
                    num_pad = model_config.max_cand_rules - len(cur_rule_id_input_placeholder)
                    cur_rule_id_input_placeholder.extend(num_pad * [0])
                    cur_rule_target_input_placeholder.extend(num_pad * pad_id)
                else:
                    cur_rule_id_input_placeholder = cur_rule_id_input_placeholder[:model_config.max_cand_rules]
                    cur_rule_target_input_placeholder = cur_rule_target_input_placeholder[:model_config.max_cand_rules]

                tmp_sups['rule_id_input_placeholder'].append(cur_rule_id_input_placeholder)
                tmp_sups['rule_target_input_placeholder'].append(cur_rule_target_input_placeholder)

            if model_config.pointer_mode == 'ptr':
                oov = obj_data['oov']
                for wd in oov['i2w']:
                    if wd not in batch_oov['w2i']:
                        batch_oov['w2i'][wd] = len(batch_oov['i2w']) + data.vocab_simple.vocab_size()
                        batch_oov['i2w'].append(wd)

        if model_config.pointer_mode == 'ptr':
            tmp_sentence_complex_ext = deepcopy(tmp_sentence_complex)
            for batch_idx in range(len(tmp_sentence_complex_ext)):
                for wid, word_comp in enumerate(tmp_sentence_complex_ext[batch_idx]):
                    if data.vocab_complex.encode(constant.SYMBOL_UNK) == word_comp:
                        tmp_sentence_complex_ext[
                            batch_idx][wid] = batch_oov['w2i'][tmp_sentence_complex_raw[batch_idx][wid]]

            tmp_sentence_simple_ext = deepcopy(tmp_sentence_simple)
            for batch_idx in range(len(tmp_sentence_simple_ext)):
                for wid, word_simp in enumerate(tmp_sentence_simple_ext[batch_idx]):
                    if data.vocab_simple.encode(constant.SYMBOL_UNK) == word_simp and tmp_sentence_simple_raw[batch_idx][wid] in batch_oov['w2i']:
                        tmp_sentence_simple_ext[
                            batch_idx][wid] = batch_oov['w2i'][tmp_sentence_simple_raw[batch_idx][wid]]

            input_feed[obj['max_oov'].name] = len(batch_oov['i2w'])

            for step in range(model_config.max_complex_sentence):
                input_feed[obj['sentence_complex_input_ext_placeholder'][step].name] = [
                    tmp_sentence_complex_ext[batch_idx][step]
                    for batch_idx in range(model_config.batch_size)]

            for step in range(model_config.max_simple_sentence):
                input_feed[obj['sentence_simple_input_ext_placeholder'][step].name] = [
                    tmp_sentence_simple_ext[batch_idx][step]
                    for batch_idx in range(model_config.batch_size)]

        for step in range(model_config.max_simple_sentence):
            input_feed[obj['sentence_simple_input_placeholder'][step].name] = [tmp_sentence_simple[batch_idx][step]
                                                            for batch_idx in range(model_config.batch_size)]
        for step in range(model_config.max_complex_sentence):
            input_feed[obj['sentence_complex_input_placeholder'][step].name] = [tmp_sentence_complex[batch_idx][step]
                                                             for batch_idx in range(model_config.batch_size)]
        input_feed[obj['sentence_idxs'].name] = [tmp_idxs[batch_idx] for batch_idx in range(model_config.batch_size)]

        if model_config.memory == 'rule':
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
    data = TrainData(model_config)

    graph = None
    if model_config.framework == 'transformer':
        graph = TransformerGraph(data, True, model_config)
    else:
        raise NotImplementedError('Unknown Framework.')

    graph.create_model_multigpu()

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
        # Handling missing vars by ourselves
        available_vars = {}
        reader = tf.train.NewCheckpointReader(ckpt_path)
        var_dict = {var.op.name: var for var in var_list}
        for var in var_dict:
            if 'global_step' in var:
                continue
            if 'optimization' in var:
                continue
            if reader.has_tensor(var):
                var_ckpt = reader.get_tensor(var)
                var_cur = var_dict[var]
                if any([var_cur.shape[i] != var_ckpt.shape[i] for i in range(len(var_ckpt.shape))]):
                    print('Variable %s missing due to shape.', var)
                else:
                    available_vars[var] = var_dict[var]
            else:
                print('Variable %s missing.', var)

        partial_restore_ckpt = slim.assign_from_checkpoint_fn(
            ckpt_path, available_vars,
            ignore_missing_vars=False, reshape_variables=False)

    def init_fn(session):
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
    start_time = datetime.now()
    while True:
        input_feed = get_graph_train_data(
            data,
            graph.objs,
            model_config)

        fetches = [graph.train_op, graph.loss, graph.global_step,
                   graph.perplexity, graph.ops, graph.xxx]
        _, loss, step, perplexity, _, xxx = sess.run(fetches, input_feed)
        perplexitys.append(perplexity)

        if step % model_config.model_print_freq == 0:
            end_time = datetime.now()
            time_span = end_time - start_time
            start_time = end_time
            print('Perplexity:\t%f at step %d using %s.' % (perplexity, step, time_span))
            perplexitys.clear()

        # if step % 20 == 0:
        #     import tracemalloc
        #
        #     tracemalloc.start()
        #
        #     # ... run your application ...
        #
        #     snapshot = tracemalloc.take_snapshot()
        #     top_stats = snapshot.statistics('lineno')
        #
        #     print("[ Top 10 ]")
        #     for stat in top_stats[:10]:
        #         print(stat)

        if model_config.model_eval_freq > 0 and step % model_config.model_eval_freq == 0:
            from model.model_config import SubValWikiEightRefConfig, SubTestWikiEightRefConfig
            from model.model_config import SubValWikiEightRefPPDBConfig, SubTestWikiEightRefPPDBConfig
            from model.model_config import DefaultTestConfig, DefaultTestConfig2
            from model.model_config import SubTestWikiSmallConfig, SubTestWikiSmallPPDBConfig
            from model.model_config import SubTestWikiEightRefConfigV2, SubTestWikiEightRefConfigV2Sing
            from model.model_config import SubTestWikiEightRefPPDBConfigV2, SubTestWikiEightRefPPDBConfigV2Sing
            from model.model_config import WikiTransValCfg, WikiTransTestCfg
            ckpt = get_ckpt(model_config.modeldir, model_config.logdir)
            print("==============================Before Eval Stat==============================")
            print_cpu_memory()
            print_gpu_memory()
            if ckpt:
                if args.mode == 'dummy':
                    eval(DefaultTestConfig(), ckpt)
                    # eval(DefaultTestConfig2(), ckpt)
                elif args.mode == 'dress' or args.mode == 'all' :
                    eval(SubValWikiEightRefConfig(), ckpt)
                    # eval(SubTestWikiEightRefConfig(), ckpt)
                    # eval(SubTestWikiEightRefConfigV2(), ckpt)
                    # eval(SubTestWikiEightRefConfigV2Sing(), ckpt)
                    # eval(SubValWikiEightRefPPDBConfigConfig(), ckpt)
                    # eval(SubTestWikiEightRefPPDBConfig(), ckpt)
                    # eval(SubTestWikiEightRefPPDBConfigV2(), ckpt)
                    # eval(SubTestWikiEightRefPPDBConfigV2Sing(), ckpt)
                elif args.mode == 'dress2':
                    eval(SubTestWikiSmallConfig(), ckpt)
                    eval(SubTestWikiSmallPPDBConfig(), ckpt)
                elif args.mode == 'wiki':
                    # eval(WikiTransValCfg(), ckpt)
                    eval(WikiTransValCfg(), ckpt)
                    eval(WikiTransTestCfg(), ckpt)
                elif args.mode == 'wikilegacy':
                    eval(WikiTransLegacyTestCfg(), ckpt)
                print("==============================After Eval Stat==============================")
                print_cpu_memory()
                print_gpu_memory()

if __name__ == '__main__':
    config = None
    if args.mode == 'dummy':
        config = DefaultTrainConfig()
    elif args.mode == 'dress':
        config = WikiDressLargeTrainConfig()
    elif args.mode == 'dressnew':
        config = WikiDressLargeNewTrainDefault()
    elif args.mode == 'dress2':
        config = WikiDressSmallTrainConfig()
    elif args.mode == 'wiki':
        config = WikiTransTrainCfg()
    elif args.mode == 'wikilegacy':
        config = WikiTransLegacyTrainCfg()
    print(list_config(config))
    train(config)
