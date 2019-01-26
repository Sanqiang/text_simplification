from util import constant


def decode_to_output(target, sentence_simple, sentence_complex, effective_batch_size,
                     ibleus=None, targets_raw=None, sentence_complex_raw=None,
                     saris=None, fkgls=None, ref_raw_lines=None, model_config=None):
    """Generate Decode Output for human read (Aggregate result together)."""
    output = ''
    for batch_i in range(effective_batch_size):
        target_batch = 'output=' + ' '.join(target[batch_i])
        sentence_simple_batch ='gt_simple=' + ' '.join(sentence_simple[batch_i])

        sentence_ref = []
        if ref_raw_lines:
            for ref_i in range(model_config.num_refs):
                sentence_ref.append('gt_ref_' + str(ref_i) + '=' + ref_raw_lines[ref_i][batch_i])
        sentence_ref = '\n'.join(sentence_ref)

        sentence_complex_batch = 'gt_complex='+ ' '.join(sentence_complex[batch_i])

        batch_targets_raw = ''
        if targets_raw is not None:
            batch_targets_raw = 'output_pp=' + ' '.join(targets_raw[batch_i])

        batch_sentence_complex_raw = ''
        if sentence_complex_raw is not None:
            batch_sentence_complex_raw = 'gt_complex_raw=' + ' '.join(sentence_complex_raw[batch_i])

        batch_ibleu = ''
        if ibleus is not None:
            batch_ibleu = 'iBLEU=' + str(ibleus[batch_i])

        batch_sari = ''
        if saris is not None:
            batch_sari = 'SARI=' + str(saris[batch_i])

        batch_fkgl = ''
        if fkgls is not None:
            batch_fkgl = 'FKGL=' + str(fkgls[batch_i])


        output_list = [target_batch, sentence_simple_batch, sentence_complex_batch,
                                  batch_ibleu, batch_sari, batch_fkgl, batch_targets_raw, batch_sentence_complex_raw]
        output_list = [entry for entry in output_list if len(entry) > 0]
        output_list.append('')
        output_list.append('')

        if sentence_ref:
            output_list.insert(2, sentence_ref)
        output_batch = '\n'.join(output_list)
        output = '\n'.join([output, output_batch])
    return output


def decode(target, voc, use_subword=False):
    target = list(target)
    batch_size = len(target)
    decode_results = []
    for i in range(batch_size):
        if use_subword:
            decode_result = voc.describe(target[i]).split(' ')
        else:
            decode_result = list(map(voc.describe, target[i]))

        # decode_result = truncate_sent(decode_result)
        decode_results.append(decode_result)
    return decode_results


def truncate_sents(decode_results):
    ndecode_results = []
    for decode_result in decode_results:
        decode_result = truncate_sent(decode_result)
        ndecode_results.append(decode_result)
    return ndecode_results


def truncate_sent(decode_result):
    if constant.SYMBOL_END in decode_result:
        eos = decode_result.index(constant.SYMBOL_END)
        decode_result = decode_result[:eos]
    s_i = 0
    if len(decode_result) > 0 and decode_result[s_i] == constant.SYMBOL_START:
        while s_i+1 < len(decode_result) and decode_result[s_i+1] == constant.SYMBOL_START:
            s_i += 1
        decode_result = decode_result[s_i+1:]
    return decode_result


def get_exclude_list(effective_batch_size, batch_size):
    """Get the list of indexs need to eclude(All <go>)."""
    exclude_idxs = []
    for i in range(batch_size):
        if i >= effective_batch_size:
            exclude_idxs.append(i)
    return exclude_idxs


def exclude_list(results, exclude_idxs):
    nresults = []
    for re_id, result in enumerate(results):
        if re_id not in exclude_idxs:
            nresults.append(result)
    return nresults
