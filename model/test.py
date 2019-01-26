# For fix slurm cannot load PYTHONPATH
import sys
# sys.path.insert(0, '/ihome/hdaqing/saz31/sanqiang/text_simplification')
# sys.path.insert(0,'/ihome/hdaqing/saz31/ts/text_simplification')
sys.path.insert(0,'/ihome/hdaqing/saz31/ts_0924/text_simplification')


from data_generator.val_data import ValData
from model.transformer import TransformerGraph
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
from os import makedirs
import math
import numpy as np
import time
from util.arguments import get_args
from model.eval import get_graph_val_data


args = get_args()


def test(model_config=None, ckpt=None):
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

    def init_fn(session):
        graph.saver.restore(session, ckpt)
        print('Restore ckpt:%s.' % ckpt)

    sv = tf.train.Supervisor(init_fn=init_fn)
    sess = sv.PrepareSession(config=session.get_session_config(model_config))
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

            target = decode(target, val_data.vocab_simple, model_config.subword_vocab_size>0)
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
            sentence_simple = decode(sentence_simple, val_data.vocab_simple, model_config.subword_vocab_size>0)
            sentence_complex = decode(sentence_complex, val_data.vocab_complex, model_config.subword_vocab_size>0)

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


if __name__ == '__main__':
    # from model.model_config import WikiDressLargeDefault, SubTestWikiEightRefConfigV2Sing, SubTestWikiEightRefPPDBConfigV2Sing, SubTestWikiEightRefPPDBConfigV2, SubTestWikiEightRefConfigV2
    # from model.model_config import SubValWikiEightRefConfig, SubTestWikiEightRefConfig, SubTestWikiEightRefPPDBConfig
    # from model.model_config import SubValWikiEightRefConfigBeam4, SubTestWikiEightRefConfigBeam4, SubTestWikiSmallPPDBConfig
    from model.model_config import WikiTransTestConfig

    ckpt = args.test_ckpt
    config = WikiTransTestConfig()
    for style in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        config.tune_style = style
        config.resultdir = '/ihome/hdaqing/saz31/tmp/tune%s/' % style
        test(config, ckpt)
    # test(SubTestWikiSmallPPDBConfig(), ckpt)
    # test(SubTestWikiEightRefConfig(), ckpt)
    # test(SubTestWikiEightRefPPDBConfig(), ckpt)

    # test(SubTestWikiEightRefConfigV2(), ckpt)
    # test(SubTestWikiEightRefPPDBConfigV2(), ckpt)
    # test(SubTestWikiEightRefConfigV2Sing(), ckpt)
    # test(SubTestWikiEightRefPPDBConfigV2Sing(), ckpt)
    # test(SubTestWikiEightRefPPDBConfig(), ckpt)
    # test(SubTestWikiEightRefConfigBeam4(), ckpt)