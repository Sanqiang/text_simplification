from data_generator.vocab import Vocab
from nltk import word_tokenize
from util import constant
from collections import defaultdict

freq = None
def populate_freq(path):
    global freq
    if freq:
        return freq

    print('Populate Freq')
    freq = defaultdict(int)
    freq[constant.SYMBOL_PAD] = 99999
    freq[constant.SYMBOL_START] = 99999
    freq[constant.SYMBOL_END] = 99999
    freq[constant.SYMBOL_GO] = 99999
    freq[constant.SYMBOL_UNK] = 99999
    for line in open(path):
        items = line.strip().split('\t')
        w = items[0]
        cnt = int(items[1])
        freq[w] = cnt

def get_segment_copy_idx(ids, freq, vocab, base_line=None):
    if base_line is not None:
        base_line = set(str(base_line).split())
    buffer = []
    idxs = []
    for id in ids:
        subtok = vocab.subword.all_subtoken_strings[id]
        buffer.append(subtok)
        if subtok.endswith('_'):
            token = ''.join(buffer)[:-1]
            if freq[token] <= 100 and not subtok.endswith(';_') and (
                    base_line is None or token in base_line):
                idxs.extend([1] * len(buffer))
            else:
                idxs.extend([0] * len(buffer))
            buffer = []
    if buffer:
        idxs.extend([0] * len(buffer))
    assert len(idxs) == len(ids)
    return idxs


def get_segment_idx(ids, vocab):
    """For token segment idx for subtoken scenario"""
    idx = 1
    idxs = []
    for id in ids:
        if id == vocab.encode(constant.SYMBOL_PAD)[0]:
            idxs.append(0)
        else:
            idxs.append(idx)
            if vocab.subword.all_subtoken_strings[id].endswith('_'):
                idx += 1
    assert len(idxs) == len(ids)
    return idxs

def process_line(line, vocab, max_len, model_config, need_raw=False, lower_case=True,
                 base_line=None):
    if lower_case:
        line = line.lower()
    if type(line) == bytes:
        line = str(line, 'utf-8')

    if model_config.tokenizer == 'split':
        words = line.split()
    elif model_config.tokenizer == 'nltk':
        words = word_tokenize(line)
    else:
        raise Exception('Unknown tokenizer.')

    words = [Vocab.process_word(word, model_config)
             for word in words]
    if need_raw:
        words_raw = [constant.SYMBOL_START] + words + [constant.SYMBOL_END]
    else:
        words_raw = None

    if model_config.subword_vocab_size > 0 or 'bert_token' in model_config.bert_mode:
        words = [constant.SYMBOL_START] + words + [constant.SYMBOL_END]
        words = vocab.encode(' '.join(words))
    else:
        words = [vocab.encode(word) for word in words]
        words = ([vocab.encode(constant.SYMBOL_START)] + words +
                 [vocab.encode(constant.SYMBOL_END)])

    if model_config.subword_vocab_size > 0 or 'bert_token' in model_config.bert_mode:
        pad_id = vocab.encode(constant.SYMBOL_PAD)
    else:
        pad_id = [vocab.encode(constant.SYMBOL_PAD)]

    if len(words) < max_len:
        num_pad = max_len - len(words)
        words.extend(num_pad * pad_id)
    else:
        words = words[:max_len]

    obj = {}
    if model_config.subword_vocab_size and 'seg' in model_config.seg_mode:
        obj['segment_idxs'] = get_segment_idx(words, vocab)
    elif model_config.subword_vocab_size and 'cp' in model_config.seg_mode:
        populate_freq('/zfs1/hdaqing/saz31/dataset/vocab/all.vocab')
        obj['segment_idxs'] = get_segment_copy_idx(words, freq, vocab,
                                                   base_line=base_line)

    return words, words_raw, obj