from data_generator.val_data import ValData
from model.transformer import TransformerGraph
from model.seq2seq import Seq2SeqGraph
from model.model_config import DefaultConfig, DefaultTestConfig, WikiDressLargeTestConfig, list_config
from data_generator.vocab import Vocab
from util import constant
from util import session
from util.checkpoint import copy_ckpt_to_modeldir
from util.decode import decode, decode_to_output, exclude_list, get_exclude_list, truncate_sents
from model.postprocess import PostProcess

from nltk.translate.bleu_score import sentence_bleu
from util.mteval_bleu import MtEval_BLEU
import tensorflow as tf
import math
import numpy as np
import time


def get_graph_val_data(sentence_simple_input, sentence_complex_input,
                       model_config, it):
    input_feed = {}
    voc = Vocab(model_config)

    tmp_sentence_simple, tmp_sentence_complex, tmp_sentence_complex_raw = [], [], []
    tmp_mapper = []
    tmp_ref = [[] for _ in range(model_config.num_refs)]
    effective_batch_size = 0
    is_end = False
    for i in range(model_config.batch_size):
        if not is_end:
            sentence_simple, sentence_complex, sentence_complex_raw, mapper, ref = next(it)
            effective_batch_size += 1
        if sentence_simple is None or is_end:
            # End of data set
            if not is_end:
                effective_batch_size -= 1
            is_end = True
            sentence_simple, sentence_complex, ref = [], [], []

        if ref:
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
        tmp_mapper.append(mapper)
        tmp_sentence_complex_raw.append(sentence_complex_raw)

    for step in range(model_config.max_simple_sentence):
        input_feed[sentence_simple_input[step].name] = [tmp_sentence_simple[batch_idx][step]
                                                        for batch_idx in range(model_config.batch_size)]
    for step in range(model_config.max_complex_sentence):
        input_feed[sentence_complex_input[step].name] = [tmp_sentence_complex[batch_idx][step]
                                                         for batch_idx in range(model_config.batch_size)]

    return (input_feed, tmp_sentence_simple, tmp_sentence_complex,
            tmp_sentence_complex_raw, tmp_mapper, tmp_ref, effective_batch_size, is_end)


def eval(model_config=None):
    model_config = (DefaultConfig()
                    if model_config is None else model_config)
    val_data = ValData(model_config)
    if model_config.framework == 'transformer':
        graph = TransformerGraph(val_data, False, model_config)
    elif model_config.framework == 'seq2seq':
        graph = Seq2SeqGraph(val_data, False, model_config)
    graph.create_model()

    while True:
        ibleus_all = []
        perplexitys_all = []
        decode_outputs_all = []
        targets = []
        targets_raw = []
        sentence_simples = []
        sentence_complexs = []
        sentence_complexs_raw = []
        refs = [[] for _ in range(model_config.num_refs)]

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
            (input_feed, sentence_simple, sentence_complex,
             sentence_complex_raw, mapper, ref, effective_batch_size, is_end) = get_graph_val_data(
                graph.sentence_simple_input_placeholder,
                graph.sentence_complex_input_placeholder,
                model_config, it)

            fetches = {'decoder_target_list': graph.decoder_target_list,
                       'loss': graph.loss,
                       'global_step': graph.global_step}
            if model_config.replace_unk_by_emb:
                fetches.update({'encoder_embs': graph.encoder_embs, 'decoder_output_list': graph.decoder_output_list})
            results = sess.run(fetches, input_feed)
            target, loss, step = results['decoder_target_list'], results['loss'], results['global_step']
            if model_config.replace_unk_by_emb:
                encoder_embs, decoder_outputs = results['encoder_embs'], results['decoder_output_list']
            batch_perplexity = math.exp(loss)
            perplexitys_all.append(batch_perplexity)

            exclude_idxs = get_exclude_list(sentence_complex, val_data.vocab_complex)
            if exclude_idxs:
                sentence_complex = exclude_list(sentence_complex, exclude_idxs)
                sentence_simple = exclude_list(sentence_simple, exclude_idxs)
                target = exclude_list(target, exclude_idxs)
                sentence_complex_raw = exclude_list(sentence_complex_raw, exclude_idxs)
                for ref_i in range(model_config.num_refs):
                    ref[ref_i] = exclude_list(ref[ref_i], exclude_idxs)


            target = decode(target, val_data.vocab_simple)
            postprocess = PostProcess(model_config, val_data)
            if model_config.replace_unk_by_emb:
                target_raw = postprocess.replace_unk_by_emb(sentence_complex_raw, encoder_embs, decoder_outputs, target)
            if model_config.replace_unk_by_cnt:
                target_raw = postprocess.replace_unk_by_cnt(sentence_complex_raw, target)
            if model_config.replace_ner:
                target_raw = postprocess.replace_ner(target, mapper)
            sentence_simple = decode(sentence_simple, val_data.vocab_simple)
            sentence_complex = decode(sentence_complex, val_data.vocab_complex)
            sentence_complex_raw = truncate_sents(sentence_complex_raw)
            for ref_i in range(model_config.num_refs):
                ref[ref_i] = decode(ref[ref_i], val_data.vocab_simple)
            targets.extend(target)
            targets_raw.extend(target_raw)
            sentence_simples.extend(sentence_simple)
            sentence_complexs.extend(sentence_complex)
            sentence_complexs_raw.extend(sentence_complex_raw)
            for ref_i in range(model_config.num_refs):
                refs[ref_i].extend(ref[ref_i])

            ibleus = []
            for batch_i in range(effective_batch_size):
                # Compute iBLEU
                try:
                    batch_bleu_i = sentence_bleu([sentence_simple[batch_i]], target[batch_i])
                    batch_bleu_rs = []
                    for ref_i in range(model_config.num_refs):
                        batch_bleu_rs.append(
                            # Note: default sentence_bleu weight unigram, bigram, trigram, quagram equally,
                            # i.e. [.25, .25, .25, .25]
                            sentence_bleu([ref[ref_i][batch_i]], target[batch_i]))
                    if len(batch_bleu_rs) > 0:
                        batch_bleu_r = max(batch_bleu_rs)
                        batch_ibleu = batch_bleu_r * 0.9 + batch_bleu_i * 0.1
                    else:
                        batch_ibleu = batch_bleu_i
                except Exception as e:
                    print('Bleu exception:\t' + str(e))
                    batch_ibleu = 0
                ibleus_all.append(batch_ibleu)
                ibleus.append(batch_ibleu)
                # print('Batch iBLEU: \t%f.' % batch_ibleu)
            target_output = decode_to_output(target, sentence_simple, sentence_complex,
                                             effective_batch_size, ibleus, target_raw, sentence_complex_raw)
            decode_outputs_all.append(target_output)

            if is_end:
                break

        ibleu = np.mean(ibleus_all)
        perplexity = np.mean(perplexitys_all)
        print('Current iBLEU: \t%f' % ibleu)
        print('Current perplexity: \t%f' % perplexity)
        print('Current eval done!')
        # MtEval Result
        mteval = MtEval_BLEU(model_config)
        # MtEval Result - Decode
        bleu_oi_decode = mteval.get_bleu_from_decoderesult(step, sentence_complexs, sentence_simples, targets)
        bleu_ors_decode = []
        for ref_i in range(model_config.num_refs):
            bleu_or_decode = mteval.get_bleu_from_decoderesult(step, sentence_complexs, refs[ref_i], targets)
            bleu_ors_decode.append(bleu_or_decode)
        bleu_decode = 0.9 * max(bleu_ors_decode) + 0.1 * bleu_oi_decode
        print('Current Mteval iBLEU decode: \t%f' % bleu_decode)

        # MtEval Result - raw
        bleu_oi_raw = mteval.get_bleu_from_rawresult(step, targets_raw)
        bleu_ors_raw = []
        for ref_i in range(model_config.num_refs):
            bleu_or_raw = mteval.get_bleu_from_rawresult(
                step, targets_raw, path_gt_simple=(model_config.val_dataset_simple_folder +
                                         model_config.val_dataset_simple_references + str(ref_i)))
            bleu_ors_raw.append(bleu_or_raw)
        bleu_raw = 0.9 * max(bleu_ors_raw) + 0.1 * bleu_oi_raw
        print('Current Mteval iBLEU decode: \t%f' % bleu_raw)


        # Output Result
        f = open((model_config.modeldir + '/step' + str(step) +
                  '-ibleu_raw' + str(bleu_raw) +
                  '-ibleu_decode' + str(bleu_decode) +
                  '-ibleu' + str(ibleu)),
                 'w', encoding='utf-8')
        f.write(str(ibleu))
        f.write('\t')
        f.write(str(perplexity))
        f.close()
        f = open((model_config.modeldir + '/step' + str(step) +
                  '-ibleu_raw' + str(bleu_raw) +
                  '-ibleu_decode' + str(bleu_decode) +
                  '-ibleu' + str(ibleu) + '.result'),
                 'w', encoding='utf-8')
        f.write('\n'.join(decode_outputs_all))
        f.close()


if __name__ == '__main__':
    config = WikiDressLargeTestConfig()
    print(list_config(config))
    eval(config)
