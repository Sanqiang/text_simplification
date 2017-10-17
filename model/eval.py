# For fix slurm cannot load PYTHONPATH
import sys
sys.path.insert(0, '/ihome/hdaqing/saz31/sanqiang/text_simplification')


from data_generator.val_data import ValData
from model.transformer import TransformerGraph
from model.seq2seq import Seq2SeqGraph
from model.model_config import DefaultConfig, DefaultTestConfig, DefaultTestConfig2, list_config
from data_generator.vocab import Vocab
from util import constant
from util import session
from util.checkpoint import copy_ckpt_to_modeldir
from util.sari import SARIsent
from util.fkgl import get_fkgl
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


args = get_args()


def get_graph_val_data(sentence_simple_input, sentence_complex_input,
                       model_config, it):
    input_feed = {}
    voc = Vocab(model_config)

    (tmp_sentence_simple, tmp_sentence_complex,
     tmp_sentence_complex_raw, tmp_sentence_complex_raw_lines) = [], [], [], []
    tmp_mapper = []
    tmp_ref = [[] for _ in range(model_config.num_refs)]
    tmp_ref_raw = [[] for _ in range(model_config.num_refs)]
    tmp_ref_raw_lines = [[] for _ in range(model_config.num_refs)]
    effective_batch_size = 0
    is_end = False
    for i in range(model_config.batch_size):
        if not is_end:
            (sentence_simple, sentence_complex, sentence_complex_raw, sentence_complex_raw_lines,
             mapper, ref, ref_raw, ref_raw_lines) = next(it)
            effective_batch_size += 1
        if sentence_simple is None or is_end:
            # End of data set
            if not is_end:
                effective_batch_size -= 1
            is_end = True
            sentence_simple, sentence_complex, ref = [], [], []

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
        tmp_mapper.append(mapper)
        tmp_sentence_complex_raw.append(sentence_complex_raw)
        tmp_sentence_complex_raw_lines.append(sentence_complex_raw_lines)
        if ref:
            for i_ref in range(model_config.num_refs):
                tmp_ref[i_ref].append(ref[i_ref])
                tmp_ref_raw[i_ref].append(ref_raw[i_ref])
                tmp_ref_raw_lines[i_ref].append(ref_raw_lines[i_ref])

    for step in range(model_config.max_simple_sentence):
        input_feed[sentence_simple_input[step].name] = [tmp_sentence_simple[batch_idx][step]
                                                        for batch_idx in range(model_config.batch_size)]
    for step in range(model_config.max_complex_sentence):
        input_feed[sentence_complex_input[step].name] = [tmp_sentence_complex[batch_idx][step]
                                                         for batch_idx in range(model_config.batch_size)]

    return (input_feed, tmp_sentence_simple,
            tmp_sentence_complex, tmp_sentence_complex_raw, tmp_sentence_complex_raw_lines,
            tmp_mapper,
            tmp_ref, tmp_ref_raw, tmp_ref_raw_lines,
            effective_batch_size,
            is_end)


def eval(model_config=None, ckpt=None):
    model_config = (DefaultConfig()
                    if model_config is None else model_config)
    if not exists(model_config.resultdor):
        makedirs(model_config.resultdor)
    print(list_config(model_config))

    val_data = ValData(model_config)
    graph = None
    if model_config.framework == 'transformer':
        graph = TransformerGraph(val_data, False, model_config)
    elif model_config.framework == 'seq2seq':
        graph = Seq2SeqGraph(val_data, False, model_config)
    tf.reset_default_graph()
    graph.create_model()

    # while True:
    ibleus_all = []
    perplexitys_all = []
    saris_all = []
    decode_outputs_all = []
    targets = []
    targets_raw = []
    sentence_simples = []
    sentence_complexs = []
    sentence_complexs_raw = []
    refs = [[] for _ in range(model_config.num_refs)]

    it = val_data.get_data_iter()

    def init_fn(session):
        graph.saver.restore(session, ckpt)
        print('Restore ckpt:%s.' % ckpt)

    sv = tf.train.Supervisor(init_fn=init_fn)
    sess = sv.PrepareSession(config=session.get_session_config(model_config))
    while True:
        (input_feed, sentence_simple,
         sentence_complex, sentence_complex_raw, sentence_complex_raw_lines,
         mapper,
         ref, ref_raw, ref_raw_lines,
         effective_batch_size, is_end) = get_graph_val_data(
            graph.sentence_simple_input_placeholder,
            graph.sentence_complex_input_placeholder,
            model_config, it)

        postprocess = PostProcess(model_config, val_data)
        # Replace UNK for sentence_complex_raw and ref_raw
        # Note that sentence_complex_raw_lines and ref_raw_lines are original file lines
        sentence_complex_raw = postprocess.replace_ner(sentence_complex_raw, mapper)
        for ref_i in range(model_config.num_refs):
            ref_raw[ref_i] = postprocess.replace_ner(ref_raw[ref_i], mapper)

        fetches = {'decoder_target_list': graph.decoder_target_list,
                   'loss': graph.loss,
                   'global_step': graph.global_step}
        if model_config.replace_unk_by_emb:
            fetches.update({'encoder_embs': graph.encoder_embs, 'decoder_outputs': graph.decoder_outputs})
        results = sess.run(fetches, input_feed)
        target, loss, step = (results['decoder_target_list'], results['loss'],
                                          results['global_step'])
        if model_config.replace_unk_by_emb:
            encoder_embs, decoder_outputs = results['encoder_embs'], results['decoder_outputs']
        batch_perplexity = math.exp(loss)
        perplexitys_all.append(batch_perplexity)

        exclude_idxs = get_exclude_list(sentence_complex, val_data.vocab_complex)
        if exclude_idxs:
            sentence_complex = exclude_list(sentence_complex, exclude_idxs)
            sentence_complex_raw = exclude_list(sentence_complex_raw, exclude_idxs)
            sentence_complex_raw_lines = exclude_list(sentence_complex_raw_lines, exclude_idxs)

            sentence_simple = exclude_list(sentence_simple, exclude_idxs)

            target = exclude_list(target, exclude_idxs)
            mapper = exclude_list(mapper, exclude_idxs)

            for ref_i in range(model_config.num_refs):
                ref[ref_i] = exclude_list(ref[ref_i], exclude_idxs)
                ref_raw[ref_i] = exclude_list(ref_raw[ref_i], exclude_idxs)
                ref_raw_lines[ref_i] = exclude_list(ref_raw_lines[ref_i], exclude_idxs)

        target = decode(target, val_data.vocab_simple)
        target_raw = target
        if model_config.replace_unk_by_attn:
            target_raw = postprocess.replace_unk_by_attn(sentence_complex_raw, None, target_raw)
        elif model_config.replace_unk_by_emb:
            target_raw = postprocess.replace_unk_by_emb(
                sentence_complex_raw, encoder_embs, decoder_outputs, target_raw)
        elif model_config.replace_unk_by_cnt:
            target_raw = postprocess.replace_unk_by_cnt(sentence_complex_raw, target_raw)
        if model_config.replace_ner:
            target_raw = postprocess.replace_ner(target_raw, mapper)
        target_raw = postprocess.replace_others(target_raw)
        sentence_simple = decode(sentence_simple, val_data.vocab_simple)
        sentence_complex = decode(sentence_complex, val_data.vocab_complex)
        sentence_complex_raw = truncate_sents(sentence_complex_raw)
        for ref_i in range(model_config.num_refs):
            ref[ref_i] = decode(ref[ref_i], val_data.vocab_simple)

        # Truncate decode results
        target = truncate_sents(target)
        target_raw = truncate_sents(target_raw)
        sentence_simple = truncate_sents(sentence_simple)
        sentence_complex = truncate_sents(sentence_complex)
        for ref_i in range(model_config.num_refs):
            ref[ref_i] = truncate_sents(ref[ref_i])

        targets.extend(target)
        targets_raw.extend(target_raw)
        sentence_simples.extend(sentence_simple)
        sentence_complexs.extend(sentence_complex)
        sentence_complexs_raw.extend(sentence_complex_raw)
        for ref_i in range(model_config.num_refs):
            refs[ref_i].extend(ref[ref_i])

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

        target_output = decode_to_output(target, sentence_simple, sentence_complex,
                                         effective_batch_size, ibleus, target_raw, sentence_complex_raw,
                                         saris, fkgls)
        decode_outputs_all.append(target_output)

        if is_end:
            break

    ibleu = np.mean(ibleus_all)
    perplexity = np.mean(perplexitys_all)
    sari = np.mean(saris_all)
    # Compute FKGL in Corpus level
    fkgl = get_fkgl(
        '\n'.join([' '.join(target_raw) for target_raw in targets_raw]))
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
    bleu_oi_raw = mteval.get_bleu_from_rawresult(step, targets_raw)
    bleu_or_raw = bleu_oi_raw
    if model_config.num_refs > 0:
        path_ref = model_config.val_dataset_simple_folder + model_config.val_dataset_simple_raw_references
        bleu_or_raw = mteval.get_bleu_from_decoderesult_multirefs(step, path_ref, targets_raw,
                                                                  lowercase=model_config.lower_case)
    if model_config.num_refs > 0:
        bleu_raw = 0.9 * bleu_or_raw + 0.1 * bleu_oi_raw
    else:
        bleu_raw = bleu_oi_raw
    print('Current Mteval iBLEU raw: \t%f' % bleu_raw)

    bleu_joshua = bleu_oi_raw
    if model_config.num_refs > 0:
        bleu_joshua = mteval.get_bleu_from_joshua(
            step, model_config.val_dataset_simple_folder + model_config.val_dataset_simple_rawlines_file_references,
            targets_raw)

    decimal_cnt = 5
    format = "%." + str(decimal_cnt) + "f"
    bleu_raw = format % bleu_raw
    bleu_oi_raw = format % bleu_oi_raw
    bleu_or_raw = format % bleu_or_raw
    # bleu_decode = format % bleu_decode
    # bleu_oi_decode = format % bleu_oi_decode
    # bleu_or_decode = format % bleu_or_decode
    ibleu = format % ibleu
    bleu_joshua = format % bleu_joshua
    sari = format % sari
    fkgl = format % fkgl
    perplexity = format % perplexity

    content = '\n'.join(['bleu_raw\t' + str(bleu_raw),
                         'bleu_oi_raw\t' + str(bleu_oi_raw),
                         'bleu_or_raw\t' + str(bleu_or_raw),
                         # 'bleu_decode\t' + str(bleu_decode),
                         # 'bleu_oi_decode\t' + str(bleu_oi_decode),
                         # 'bleu_or_decode\t' + str(bleu_or_decode),
                         'ibleu\t' + str(ibleu),
                         'bleu_joshua\t' + str(bleu_joshua),
                         'sari\t' + str(sari),
                         'fkgl\t' + str(fkgl)
                         ])

    # Output Result
    f = open((model_config.resultdor + '/step' + str(step) +
              '-bleuraw' + str(bleu_raw) +
              '-bleurawoi' + str(bleu_oi_raw) +
              '-bleurawor' + str(bleu_or_raw) +
              '-bleuj' + str(bleu_joshua) +
              '-perplexity' + str(perplexity) +
              '-bleunltk' + str(ibleu) +
              '-sari' + str(sari) +
              '-fkgl' + str(fkgl)
              ),
             'w', encoding='utf-8')
    f.write(content)
    f.close()
    f = open((model_config.resultdor + '/step' + str(step) +
              '-bleuraw' + str(bleu_raw) +
              '-bleurawoi' + str(bleu_oi_raw) +
              '-bleurawor' + str(bleu_or_raw) +
              '-bleuj' + str(bleu_joshua) +
              '-perplexity' + str(perplexity) +
              '-bleunltk' + str(ibleu) +
              '-sari' + str(sari) +
              '-fkgl' + str(fkgl)+ '.result'),
             'w', encoding='utf-8')
    f.write('\n'.join(decode_outputs_all))
    f.close()


def get_ckpt(modeldir, logdir, wait_second=60):
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

if __name__ == '__main__':
    config = None
    if args.mode == 'dummy':
        while True:
            model_config = DefaultTestConfig()
            ckpt = get_ckpt(model_config.modeldir, model_config.logdir)
            if ckpt:
                eval(DefaultTestConfig(), ckpt)
                eval(DefaultTestConfig2(), ckpt)
    elif args.mode == 'all' or args.mode == 'dress':
        from model.model_config import WikiDressLargeDefault
        from model.model_config import SubValWikiEightRefConfig, SubTestWikiEightRefConfig
        from model.model_config import SubValWikiDress, SubTestWikiDress
        from model.model_config import SubValWikiDressBeam4, SubTestWikiDressBeam4
        from model.model_config import SubValWikiEightRefConfigBeam4, SubTestWikiEightRefConfigBeam4
        from model.model_config import SubValWikiDressL, SubTestWikiDressL
        from model.model_config import SubValWikiDressLBeam4, SubTestWikiDressLBeam4
        while True:
            model_config = WikiDressLargeDefault()
            ckpt = get_ckpt(model_config.modeldir, model_config.logdir)

            if ckpt:
                eval(SubValWikiEightRefConfig(), ckpt)
                eval(SubTestWikiEightRefConfig(), ckpt)

                eval(SubValWikiDressL(), ckpt)
                eval(SubTestWikiDressL(), ckpt)

                eval(SubValWikiDress(), ckpt)
                eval(SubTestWikiDress(), ckpt)

                eval(SubValWikiDressLBeam4(), ckpt)
                eval(SubTestWikiDressLBeam4(), ckpt)

                eval(SubValWikiEightRefConfigBeam4(), ckpt)
                eval(SubTestWikiEightRefConfigBeam4(), ckpt)

                eval(SubValWikiDressBeam4(), ckpt)
                eval(SubTestWikiDressBeam4(), ckpt)

