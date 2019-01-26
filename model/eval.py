# For fix slurm cannot load PYTHONPATH
import sys
# sys.path.insert(0, '/ihome/hdaqing/saz31/sanqiang/text_simplification')
# sys.path.insert(0,'/home/hed/text_simp_proj/text_simplification')
# sys.path.insert(0,'/ihome/cs2770_s2018/maz54/ts/text_simplification')
# sys.path.insert(0,'/ihome/hdaqing/saz31/ts/text_simplification')
sys.path.insert(0,'/ihome/hdaqing/saz31/ts_0924/text_simplification')


from data_generator.val_data import ValData
from model.transformer import TransformerGraph
from model.seq2seq import Seq2SeqGraph
from model.model_config import DefaultConfig, DefaultTestConfig, list_config
from data_generator.vocab import Vocab
from util import constant
from util import session
from util.checkpoint import copy_ckpt_to_modeldir
from util.sari import SARIsent
from util.sari import CorpusSARI
from util.fkgl import get_fkgl, CorpusFKGL
from util.decode import decode, decode_to_output, exclude_list, get_exclude_list, truncate_sents
from model.postprocess import PostProcess

from nltk.translate.bleu_score import sentence_bleu
from util.mteval_bleu import MtEval_BLEU
import tensorflow as tf
from os.path import exists
from os import remove,listdir,makedirs
import math
import numpy as np
import time
import glob
from util.arguments import get_args
from copy import deepcopy
from model.model_config import get_path


args = get_args()


def get_graph_val_data(objs,
                       model_config, it, data):
    input_feed = {}
    # Reserved section of vocabuary are same.
    voc = data.vocab_simple

    if model_config.subword_vocab_size > 0 or 'bert_token' in model_config.bert_mode:
        pad_id = voc.encode(constant.SYMBOL_PAD)
    else:
        pad_id = [voc.encode(constant.SYMBOL_PAD)]

    output_tmp_sentence_simple, output_tmp_sentence_complex, \
    output_tmp_sentence_complex_raw, output_tmp_sentence_complex_raw_lines, \
    output_tmp_mapper, output_tmp_ref_raw_lines = [], [], [], [], [], []
    output_effective_batch_size, output_is_end = [], []

    for obj in objs:

        (tmp_sentence_simple, tmp_sentence_complex,
         tmp_sentence_complex_raw, tmp_sentence_complex_raw_lines, tmp_sups, tmp_sentence_simple_raw) = [], [], [], [], {}, []
        tmp_mapper = []
        tmp_ref_raw_lines = [[] for _ in range(model_config.num_refs)]
        effective_batch_size = 0
        is_end = False
        for i in range(model_config.batch_size):
            if not is_end:
                obj_data = next(it)
                effective_batch_size += 1
            if obj_data is None or is_end:
                # End of data set
                if not is_end:
                    effective_batch_size -= 1
                is_end = True

            if obj_data is not None:
                for i_ref in range(model_config.num_refs):
                    tmp_ref_raw_lines[i_ref].append(obj_data['ref_raw_lines'][i_ref])

                tmp_sentence_simple.append(obj_data['sentence_simple'])
                tmp_sentence_complex.append(obj_data['sentence_complex'])
                tmp_mapper.append(obj_data['mapper'])
                tmp_sentence_complex_raw.append(obj_data['sentence_complex_raw'])
                tmp_sentence_simple_raw.append(obj_data['sentence_simple_raw'])
                tmp_sentence_complex_raw_lines.append(obj_data['sentence_complex_raw_lines'])

                if 'rule' in model_config.memory or 'direct' in model_config.memory:
                    if 'rule_id_input_placeholder' not in tmp_sups:
                        tmp_sups['rule_id_input_placeholder'] = []
                    if 'rule_target_input_placeholder' not in tmp_sups:
                        tmp_sups['rule_target_input_placeholder'] = []

                    cur_rule_id_input_placeholder = []
                    cur_rule_target_input_placeholder = []
                    sup = obj_data['sup']
                    if sup and 'mem' in sup:
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

                if model_config.tune_style:
                    if 'comp_add_score' not in tmp_sups:
                        tmp_sups['comp_add_score'] = []
                    if 'comp_length' not in tmp_sups:
                        tmp_sups['comp_length'] = []

                    sup = obj_data['sup']
                    tmp_sups['comp_add_score'].append(sup['comp_features'][0])
                    tmp_sups['comp_length'].append(sup['comp_features'][1])

                if model_config.subword_vocab_size and model_config.seg_mode:
                    if 'tmp_sentence_seg_simple' not in tmp_sups:
                        tmp_sups['tmp_sentence_seg_simple'] = []
                    if 'tmp_sentence_seg_complex' not in tmp_sups:
                        tmp_sups['tmp_sentence_seg_complex'] = []
                    tmp_sups['tmp_sentence_seg_simple'].append(obj_data['line_simp_segids'])
                    tmp_sups['tmp_sentence_seg_complex'].append(obj_data['line_comp_segids'])
            else:
                if model_config.subword_vocab_size > 0 or 'bert_token' in model_config.bert_mode:
                    tmp_sentence_simple.append(
                        data.vocab_simple.encode(constant.SYMBOL_PAD) * model_config.max_simple_sentence)
                    tmp_sentence_complex.append(
                        data.vocab_complex.encode(constant.SYMBOL_PAD) * model_config.max_complex_sentence)
                else:
                    tmp_sentence_simple.append(
                        [data.vocab_simple.encode(constant.SYMBOL_PAD)] * model_config.max_simple_sentence)
                    tmp_sentence_complex.append(
                        [data.vocab_complex.encode(constant.SYMBOL_PAD)] * model_config.max_complex_sentence)
                if 'rule' in model_config.memory or 'direct' in model_config.memory:
                    if 'rule_id_input_placeholder' not in tmp_sups:
                        tmp_sups['rule_id_input_placeholder'] = []
                    if 'rule_target_input_placeholder' not in tmp_sups:
                        tmp_sups['rule_target_input_placeholder'] = []
                    tmp_sups['rule_id_input_placeholder'].append(
                        [0] * model_config.max_cand_rules)
                    tmp_sups['rule_target_input_placeholder'].append(
                        data.vocab_simple.encode(constant.SYMBOL_PAD) * model_config.max_cand_rules)

                if model_config.tune_style:
                    tmp_sups['comp_add_score'].append(0)
                    tmp_sups['comp_length'].append(0)

                if model_config.subword_vocab_size and model_config.seg_mode:
                    if model_config.subword_vocab_size > 0:
                        tmp_sups['tmp_sentence_seg_simple'].append(
                            data.vocab_simple.encode(constant.SYMBOL_PAD) * model_config.max_simple_sentence)
                        tmp_sups['tmp_sentence_seg_complex'].append(
                            data.vocab_complex.encode(constant.SYMBOL_PAD) * model_config.max_complex_sentence)
                    else:
                        tmp_sups['tmp_sentence_seg_simple'].append(
                            [data.vocab_simple.encode(constant.SYMBOL_PAD)] * model_config.max_simple_sentence)
                        tmp_sups['tmp_sentence_seg_complex'].append(
                            [data.vocab_complex.encode(constant.SYMBOL_PAD)] * model_config.max_complex_sentence)

        for step in range(model_config.max_simple_sentence):
            input_feed[obj['sentence_simple_input_placeholder'][step].name] = [tmp_sentence_simple[batch_idx][step]
                                                            for batch_idx in range(model_config.batch_size)]
        for step in range(model_config.max_complex_sentence):
            input_feed[obj['sentence_complex_input_placeholder'][step].name] = [tmp_sentence_complex[batch_idx][step]
                                                             for batch_idx in range(model_config.batch_size)]

        if 'rule' in model_config.memory or 'direct' in model_config.memory:
            for step in range(model_config.max_cand_rules):
                input_feed[obj['rule_id_input_placeholder'][step].name] = [
                    tmp_sups['rule_id_input_placeholder'][batch_idx][step]
                    for batch_idx in range(model_config.batch_size)]
                input_feed[obj['rule_target_input_placeholder'][step].name] = [
                    tmp_sups['rule_target_input_placeholder'][batch_idx][step]
                    for batch_idx in range(model_config.batch_size)]

        if model_config.tune_style:
            input_feed[obj['comp_features']['comp_add_score'].name] = [
                tmp_sups['comp_add_score'][batch_idx]
                for batch_idx in range(model_config.batch_size)]
            input_feed[obj['comp_features']['comp_length'].name] = [
                tmp_sups['comp_length'][batch_idx]
                for batch_idx in range(model_config.batch_size)]

        if model_config.subword_vocab_size and model_config.seg_mode:
            for step in range(model_config.max_simple_sentence):
                input_feed[obj['sentence_simple_segment_input_placeholder'][step].name] = [
                    tmp_sups['tmp_sentence_seg_simple'][batch_idx][step]
                    for batch_idx in range(model_config.batch_size)]
            for step in range(model_config.max_complex_sentence):
                input_feed[obj['sentence_complex_segment_input_placeholder'][step].name] = [
                    tmp_sups['tmp_sentence_seg_complex'][batch_idx][step]
                    for batch_idx in range(model_config.batch_size)]

        output_tmp_sentence_simple.append(tmp_sentence_simple)
        output_tmp_sentence_complex.append(tmp_sentence_complex)
        output_tmp_sentence_complex_raw.append(tmp_sentence_complex_raw)
        output_tmp_sentence_complex_raw_lines.append(tmp_sentence_complex_raw_lines)
        output_tmp_mapper.append(tmp_mapper)
        output_tmp_ref_raw_lines.append(tmp_ref_raw_lines)
        output_effective_batch_size.append(effective_batch_size)
        output_is_end.append(is_end)

    return (input_feed, output_tmp_sentence_simple,
            output_tmp_sentence_complex, output_tmp_sentence_complex_raw, output_tmp_sentence_complex_raw_lines,
            output_tmp_mapper,
            output_tmp_ref_raw_lines,
            output_effective_batch_size,
            output_is_end)


def eval(model_config=None, ckpt=None, metric='sari'):
    model_config = (DefaultConfig()
                    if model_config is None else model_config)
    if not exists(model_config.resultdir):
        makedirs(model_config.resultdir)
    print(list_config(model_config))

    val_data = ValData(model_config)
    graph = None
    if model_config.framework == 'transformer':
        graph = TransformerGraph(val_data, False, model_config)
    elif model_config.framework == 'seq2seq':
        graph = Seq2SeqGraph(val_data, False, model_config)
    tf.reset_default_graph()
    graph.create_model_multigpu()

    ibleus_all = []
    perplexitys_all = []
    saris_all = []
    decode_outputs_all = []
    targets = []
    targets_raw = []
    sentence_simples = []
    sentence_complexs = []
    sentence_complexs_raw = []

    it = val_data.get_data_iter()

    sess = tf.train.MonitoredTrainingSession(
        config=session.get_session_config(model_config),
    )
    graph.saver.restore(sess, ckpt)

    while True:
        is_finish = False
        (input_feed, output_sentence_simple,
         output_sentence_complex, output_sentence_complex_raw, output_sentence_complex_raw_lines,
         output_mapper,
         output_ref_raw_lines,
         out_effective_batch_size, output_is_end) = get_graph_val_data(graph.objs,
            model_config, it, val_data)

        postprocess = PostProcess(model_config, val_data)
        fetches = {'decoder_target_list': [obj['decoder_target_list'] for obj in graph.objs],
                   'loss': graph.loss,
                   'global_step': graph.global_step}
        if model_config.replace_unk_by_emb:
            fetches.update(
                {'encoder_embs': [obj['encoder_embs'] for obj in graph.objs],
                 'final_outputs': [obj['final_outputs'] for obj in graph.objs]})
        if model_config.replace_unk_by_attn:
            fetches.update({'attn_distr': [obj['attn_distr'] for obj in graph.objs]})
        results = sess.run(fetches, input_feed)
        output_target, loss, step = (results['decoder_target_list'], results['loss'],
                                          results['global_step'])
        if model_config.replace_unk_by_emb:
            output_encoder_embs, output_final_outputs = results['encoder_embs'], results['final_outputs']
        if model_config.replace_unk_by_attn:
            attn_distr = results['attn_distr']

        try:
            batch_perplexity = math.exp(loss)
        except OverflowError:
            batch_perplexity = 100.0

        perplexitys_all.append(batch_perplexity)

        for i, effective_batch_size in enumerate(out_effective_batch_size):
            is_end = output_is_end[i]
            exclude_idxs = get_exclude_list(effective_batch_size, model_config.batch_size)
            sentence_simple = output_sentence_simple[i]
            sentence_complex = output_sentence_complex[i]
            sentence_complex_raw = output_sentence_complex_raw[i]
            sentence_complex_raw_lines = output_sentence_complex_raw_lines[i]
            mapper = output_mapper[i]
            ref_raw_lines = output_ref_raw_lines[i]

            target = output_target[i]
            if model_config.replace_unk_by_emb:
                encoder_embs = output_encoder_embs[i]
                final_outputs = output_final_outputs[i]

            if exclude_idxs:
                sentence_complex = exclude_list(sentence_complex, exclude_idxs)
                sentence_complex_raw = exclude_list(sentence_complex_raw, exclude_idxs)
                sentence_complex_raw_lines = exclude_list(sentence_complex_raw_lines, exclude_idxs)

                sentence_simple = exclude_list(sentence_simple, exclude_idxs)

                target = exclude_list(target, exclude_idxs)
                mapper = exclude_list(mapper, exclude_idxs)

                for ref_i in range(model_config.num_refs):
                    ref_raw_lines[ref_i] = exclude_list(ref_raw_lines[ref_i], exclude_idxs)

            target = decode(target, val_data.vocab_simple,
                            model_config.subword_vocab_size>0 or 'bert_token' in model_config.bert_mode)
            target_raw = target

            sentence_complex_marker = [[val_data.vocab_simple.encode(w)==val_data.vocab_simple.encode(constant.SYMBOL_UNK)
                                 for w in sent] for sent in sentence_complex_raw]
            if model_config.replace_unk_by_attn:
                target_raw = postprocess.replace_unk_by_attn(sentence_complex_raw, attn_distr[0], target_raw)
            elif model_config.replace_unk_by_emb:
                target_raw = postprocess.replace_unk_by_emb(
                    sentence_complex_raw, encoder_embs, final_outputs, target_raw, sentence_complex_marker)
            elif model_config.replace_unk_by_cnt:
                target_raw = postprocess.replace_unk_by_cnt(sentence_complex_raw, target_raw)
            if model_config.replace_ner:
                target_raw = postprocess.replace_ner(target_raw, mapper)
            target_raw = postprocess.replace_others(target_raw)
            sentence_simple = decode(sentence_simple, val_data.vocab_simple,
                                     model_config.subword_vocab_size>0 or 'bert_token' in model_config.bert_mode)
            sentence_complex = decode(sentence_complex, val_data.vocab_complex,
                                      model_config.subword_vocab_size>0 or 'bert_token' in model_config.bert_mode)

            # Replace UNK for sentence_complex_raw and ref_raw
            # Note that sentence_complex_raw_lines and ref_raw_lines are original file lines
            sentence_complex_raw = postprocess.replace_ner(sentence_complex_raw, mapper)
            sentence_complex_raw = truncate_sents(sentence_complex_raw)

            # Truncate decode results
            target = truncate_sents(target)
            target_raw = truncate_sents(target_raw)
            sentence_simple = truncate_sents(sentence_simple)
            sentence_complex = truncate_sents(sentence_complex)

            targets.extend(target)
            targets_raw.extend(target_raw)
            sentence_simples.extend(sentence_simple)
            sentence_complexs.extend(sentence_complex)
            sentence_complexs_raw.extend(sentence_complex_raw)

            ibleus = []
            saris = []
            fkgls = []

            for batch_i in range(effective_batch_size):
                # Compute iBLEU
                try:
                    batch_ibleu = sentence_bleu([sentence_simple[batch_i]], target[batch_i])
                except Exception as e:
                    print('Bleu error:\t' + str(e) + '\n' + str(target[batch_i]) + '\n')
                    batch_ibleu = 0
                ibleus_all.append(batch_ibleu)
                ibleus.append(batch_ibleu)

                # Compute SARI
                batch_sari = 0
                if model_config.num_refs > 0:
                    rsents = []
                    for ref_i in range(model_config.num_refs):
                        rsents.append(ref_raw_lines[ref_i][batch_i])
                    try:
                        batch_sari = SARIsent(sentence_complex_raw_lines[batch_i],
                                              ' '.join(target_raw[batch_i]),
                                              rsents)
                    except:
                        print('sari error: %s \n %s \n %s. \n' %
                              (sentence_complex_raw_lines[batch_i],
                               ' '.join(target_raw[batch_i]), rsents))
                saris.append(batch_sari)
                saris_all.append(batch_sari)

                # Compute FKGL
                target_text = ' '.join(target_raw[batch_i])
                batch_fkgl = 0
                if len(target_text) > 0:
                    batch_fkgl = get_fkgl(' '.join(target_raw[batch_i]))
                fkgls.append(batch_fkgl)

            # target_output = decode_to_output(target, sentence_simple, sentence_complex,
            #                                  effective_batch_size, ibleus, target_raw, sentence_complex_raw,
            #                                  saris, fkgls)
            target_output = decode_to_output(target, sentence_simple, sentence_complex,
                                             effective_batch_size, ibleus, target_raw, sentence_complex_raw,
                                             saris, fkgls, ref_raw_lines, model_config)
            decode_outputs_all.append(target_output)

            if is_end:
                is_finish = True
                break

        if is_finish:
            break

    ibleu = np.mean(ibleus_all)
    perplexity = np.mean(perplexitys_all)
    sari = np.mean(saris_all)
    # Compute FKGL in Corpus level
    fkgl = CorpusFKGL(model_config).get_fkgl_from_joshua(step, targets_raw)

    print('Current iBLEU: \t%f' % ibleu)
    print('Current SARI: \t%f' % sari)
    print('Current FKGL: \t%f' % fkgl)
    print('Current perplexity: \t%f' % perplexity)
    print('Current eval done!')
    # MtEval Result
    mteval = MtEval_BLEU(model_config)

    # MtEval Result - Decode
    # bleu_oi_decode = mteval.get_bleu_from_decoderesult(step, sentence_complexs, sentence_simples, targets)
    # bleu_or_decode = bleu_oi_decode
    # if model_config.num_refs > 0:
    #     path_ref = model_config.val_dataset_simple_folder + model_config.val_dataset_simple_references
    #     #Decode evaluation must be lowercase because the processed files are all lowercased
    #     bleu_or_decode = mteval.get_bleu_from_decoderesult_multirefs(step, path_ref, targets,
    #                                                                  lowercase=True)
    # if model_config.num_refs > 0:
    #     bleu_decode = 0.9 * bleu_or_decode + 0.1 * bleu_oi_decode
    # else:
    #     bleu_decode = bleu_oi_decode
    # print('Current Mteval iBLEU decode: \t%f' % bleu_decode)

    # MtEval Result - raw
    # bleu_oi_raw = mteval.get_bleu_from_rawresult(step, targets_raw)
    # bleu_or_raw = bleu_oi_raw
    # if model_config.num_refs > 0:
    #     path_ref = model_config.val_dataset_simple_folder + model_config.val_dataset_simple_rawlines_file_references
    #     bleu_or_raw = mteval.get_bleu_from_decoderesult_multirefs(step, path_ref, targets_raw,
    #                                                               lowercase=model_config.lower_case)
    # if model_config.num_refs > 0:
    #     bleu_raw = 0.9 * bleu_or_raw + 0.1 * bleu_oi_raw
    # else:
    #     bleu_raw = bleu_oi_raw
    # print('Current Mteval iBLEU raw: \t%f' % bleu_raw)

    bleu_joshua = mteval.get_bleu_from_joshua(
        step, model_config.val_dataset_simple_folder + model_config.val_dataset_simple_rawlines_file,
        model_config.val_dataset_simple_folder + model_config.val_dataset_simple_rawlines_file_references,
        targets_raw)

    # Use corpus-level sari
    corpus_sari = CorpusSARI(model_config)
    sari_joshua = corpus_sari.get_sari_from_joshua(
        step, model_config.val_dataset_simple_folder + model_config.val_dataset_simple_rawlines_file,
        model_config.val_dataset_simple_folder + model_config.val_dataset_simple_rawlines_file_references,
        model_config.val_dataset_complex_rawlines_file, target_raw
    )


    decimal_cnt = 5
    format = "%." + str(decimal_cnt) + "f"
    # bleu_raw = format % bleu_raw
    # bleu_oi_raw = format % bleu_oi_raw
    # bleu_or_raw = format % bleu_or_raw
    # bleu_decode = format % bleu_decode
    # bleu_oi_decode = format % bleu_oi_decode
    # bleu_or_decode = format % bleu_or_decode
    ibleu = format % ibleu
    bleu_joshua = format % bleu_joshua
    sari_joshua = format % sari_joshua
    fkgl = format % fkgl
    perplexity = format % perplexity

    content = '\n'.join([
        # 'bleu_raw\t' + str(bleu_raw),
        # 'bleu_oi_raw\t' + str(bleu_oi_raw),
        # 'bleu_or_raw\t' + str(bleu_or_raw),
        # 'bleu_decode\t' + str(bleu_decode),
        # 'bleu_oi_decode\t' + str(bleu_oi_decode),
        # 'bleu_or_decode\t' + str(bleu_or_decode),
        'ibleu\t' + str(ibleu),
        'bleu_joshua\t' + str(bleu_joshua),
        'sari\t' + str(sari_joshua),
        'fkgl\t' + str(fkgl)
                         ])

    # Output Result
    f = open((model_config.resultdir + '/step' + str(step) +
              # '-bleuraw' + str(bleu_raw) +
              # '-bleurawoi' + str(bleu_oi_raw) +
              # '-bleurawor' + str(bleu_or_raw) +
              '-bleuj' + str(bleu_joshua) +
              # '-perplexity' + str(perplexity) +
              '-bleunltk' + str(ibleu) +
              '-sari' + str(sari_joshua) +
              '-fkgl' + str(fkgl)
              ),
             'w', encoding='utf-8')
    f.write(content)
    f.close()
    f = open((model_config.resultdir + '/step' + str(step) +
              # '-bleuraw' + str(bleu_raw) +
              # '-bleurawoi' + str(bleu_oi_raw) +
              # '-bleurawor' + str(bleu_or_raw) +
              '-bleuj' + str(bleu_joshua) +
              # '-perplexity' + str(perplexity) +
              '-bleunltk' + str(ibleu) +
              '-sari' + str(sari_joshua) +
              '-fkgl' + str(fkgl)+ '.result'),
             'w', encoding='utf-8')
    f.write('\n'.join(decode_outputs_all))
    f.close()

    sess.close()
    if metric == 'bleu_nltk':
        return bleu_joshua
    else:
        return sari_joshua


def get_ckpt(modeldir, logdir, wait_second=10):
    while True:
        try:
            ckpt = copy_ckpt_to_modeldir(modeldir, logdir)
            return ckpt
        except FileNotFoundError as exp:
            if wait_second:
                print(str(exp) + '\nWait for 1 minutes.')
                time.sleep(wait_second)
            else:
                return None


def get_best_sari(resultdir):
    best_sari = 0.0
    if exists(resultdir):
        results = listdir(resultdir)
        for result in results:
            if result.startswith('step') and result.endswith('.result'):
                sari = float(result[(result.index('sari')+len('sari')):result.rindex('-fkgl')])
                best_sari = max(sari, best_sari)
    return best_sari

if __name__ == '__main__':
    config = None

    if args.mode == 'dummy':
        while True:
            model_config = DefaultTestConfig()
            ckpt = get_ckpt(model_config.modeldir, model_config.logdir)
            if ckpt:
                eval(DefaultTestConfig(), ckpt)
                # eval(DefaultTestConfig2(), ckpt)
    elif args.mode == 'dress':
        from model.model_config import WikiDressLargeDefault, WikiDressLargeEvalDefault, WikiDressLargeTestDefault

        best_sari = None
        while True:
            model_config = WikiDressLargeDefault()
            ckpt = get_ckpt(model_config.modeldir, model_config.logdir)

            if ckpt:
                vconfig = WikiDressLargeEvalDefault()
                if best_sari is None:
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

                    tconfig = WikiDressLargeTestDefault()
                    for rcand in [15, 30, 50]:
                        tconfig.max_cand_rules = rcand
                        tconfig.resultdir = get_path(
                            '../' + tconfig.output_folder + '/result/eightref_test_cand' + str(rcand),
                            tconfig.environment)
                        eval(tconfig, ckpt)
                print('=====================Current Best SARI:%s=====================' % best_sari)
                # if float(sari_point) < best_sari:
                #     remove(ckpt + '.index')
                #     remove(ckpt + '.meta')
                #     remove(ckpt + '.data-00000-of-00001')
                #     print('remove ckpt:%s' % ckpt)
                # else:
                #     for file in listdir(model_config.modeldir):
                #         step = ckpt[ckpt.rindex('model.ckpt-') + len('model.ckpt-'):-1]
                #         if step not in file:
                #             remove(model_config.modeldir + file)
                #     print('Get Best Model, remove ckpt except:%s.' % ckpt)
                #     best_sari = float(sari_point)

    elif args.mode == 'dressnew' or args.mode == 'wikihuge':
        from model.model_config import WikiDressLargeNewEvalDefault,WikiDressLargeNewTestDefault, WikiDressLargeNewDefault, WikiDressLargeNewEvalForBatchSize
        best_sari = None
        while True:
            model_config = WikiDressLargeNewDefault()
            ckpt = get_ckpt(model_config.modeldir, model_config.logdir)

            if ckpt:
                vconfig = WikiDressLargeNewTestDefault()
                if best_sari is None:
                    best_sari = get_best_sari(vconfig.resultdir)

                sari_point = eval(vconfig, ckpt)

                # Try different max_cand_rules
                # if args.memory is not None and 'rule' in args.memory:
                #     for rcand in [15, 30, 50]:
                #         vconfig.max_cand_rules = rcand
                #         vconfig.resultdir = get_path(
                #             '../' + vconfig.output_folder + '/result/eightref_val_cand' + str(rcand),
                #             vconfig.environment)
                #         eval(vconfig, ckpt)

                eval(WikiDressLargeNewTestDefault(), ckpt)
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
                    best_sari = float(sari_point)
    elif (args.mode == 'trans' or args.mode == 'transbert' or args.mode == 'dresstk'
          or args.mode == 'wikisplit' or args.mode == 'newsela' or args.mode == 'sentcompress'
          or args.mode == 'comb' or args.mode == 'transbert_ori'):
        from model.model_config import WikiTransEvalConfig, WikiTransTestConfig
        from model.model_config import WikiTransBertEvalConfig, WikiTransBertTestConfig
        from model.model_config import WikiTransDressTokenEvalConfig, WikiTransDressTokenTestConfig
        from model.model_config import WikiSplitBertEvalConfig, WikiSplitBertTestConfig
        from model.model_config import NewselaBertEvalConfig, NewselaBertTestConfig
        from model.model_config import SentCompressBertEvalConfig
        from model.model_config import WikiOriBertEvalConfig, WikiOriBertTestConfig

        if args.mode == 'trans':
            val_model_config = WikiTransEvalConfig()
            test_model_config = WikiTransTestConfig()
        elif args.mode == 'transbert':
            val_model_config = WikiTransBertEvalConfig()
            test_model_config = WikiTransBertTestConfig()
        elif args.mode == 'dresstk':
            val_model_config = WikiTransDressTokenEvalConfig()
            test_model_config = WikiTransDressTokenTestConfig()
        elif args.mode == 'wikisplit':
            val_model_config = WikiSplitBertEvalConfig()
            test_model_config = WikiSplitBertTestConfig()
        elif args.mode == 'newsela':
            val_model_config = NewselaBertEvalConfig()
            test_model_config = NewselaBertTestConfig()
        elif args.mode == 'sentcompress':
            val_model_config = SentCompressBertEvalConfig()
            test_model_config = SentCompressBertEvalConfig()
        elif args.mode == 'transbert_ori':
            val_model_config = WikiOriBertEvalConfig()
            test_model_config = WikiOriBertTestConfig()
        else:
            raise ValueError('unknown config')

        best_sari = None
        while True:
            ckpt = get_ckpt(val_model_config.modeldir, val_model_config.logdir)

            if ckpt:
                if best_sari is None:
                    best_sari = get_best_sari(val_model_config.resultdir)

                sari_point = eval(val_model_config, ckpt)

                # eval(test_model_config, ckpt)
                for beam_search_size in [1, 4, 8, 16, 32]:
                    test_model_config.beam_search_size = beam_search_size
                    test_model_config.resultdir = get_path(
                        '../' + test_model_config.output_folder + '/result/eightref_test_b%s' % beam_search_size,
                        test_model_config.environment)
                    eval(test_model_config, ckpt)

                print('=====================Current Best SARI:%s=====================' % best_sari)
                if float(sari_point) < best_sari:
                    for fl in glob.glob(ckpt + '*'):
                        remove(fl)
                        print('remove ckpt file:%s' % fl)
                else:
                    for file in listdir(val_model_config.modeldir):
                        step = ckpt[ckpt.rindex('model.ckpt-') + len('model.ckpt-'):-1]
                        if step not in file:
                            remove(val_model_config.modeldir + file)
                    print('Get Best Model, remove ckpt except:%s.' % ckpt)
                    best_sari = float(sari_point)



