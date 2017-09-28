from util import constant


def decode_to_output(target, sentence_simple, sentence_complex, effective_batch_size, ibleus=None):
    """Generate Decode Output for human read (Aggregate result together)."""
    output = ''
    for batch_i in range(effective_batch_size):
        target_batch = 'output=' + ' '.join(target[batch_i])
        sentence_simple_batch ='gt_simple=' + ' '.join(sentence_simple[batch_i])
        sentence_complex_batch = 'gt_complex='+ ' '.join(sentence_complex[batch_i])
        batch_ibleu = ''
        if ibleus is not None:
            batch_ibleu = 'iBLEU=' + str(ibleus[batch_i])
        output_batch = '\n'.join([target_batch, sentence_simple_batch, sentence_complex_batch,
                                  batch_ibleu, ''])
        output = '\n'.join([output, output_batch])
    return output


def decode(target, voc):
    target = list(target)
    batch_size = len(target)
    decode_results = []
    for i in range(batch_size):
        decode_result = list(map(voc.describe, target[i]))
        if constant.SYMBOL_END in decode_result:
            eos = decode_result.index(constant.SYMBOL_END)
            decode_result = decode_result[:eos]
        if len(decode_result) > 0 and decode_result[0] == constant.SYMBOL_START:
            decode_result = decode_result[1:]
        decode_results.append(decode_result)
    return decode_results


def get_exclude_list(results, voc):
    """Get the list of indexs need to eclude(All <go>)."""
    exclude_idxs = []
    for re_id, result in enumerate(results):
        is_exclude = True
        for word_id in result:
            if word_id != voc.encode(constant.SYMBOL_PAD):
                is_exclude = False
                break
        if is_exclude:
            exclude_idxs.append(re_id)
    return exclude_idxs

def exclude_list(results, exclude_idxs):
    nresults = []
    for re_id, result in enumerate(results):
        if re_id not in exclude_idxs:
            nresults.append(result)
    return nresults
