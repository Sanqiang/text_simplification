import re
from queue import Queue
from util import constant


BERT_VOCAB = '/Users/zhaosanqiang916/git/ts/text_simplification_data/bert/ber_vocab.txt'
ORIG_VOCAB = '/Users/zhaosanqiang916/git/ts/text_simplification_data/bert/all.vocab'
OUTPUT_VOCAB = '/Users/zhaosanqiang916/git/ts/text_simplification_data/bert/vocab'

def prepare_vocab():
    ner_words = Queue()
    # pad is default first one in new vocab
    ner_words.put(constant.SYMBOL_START)
    ner_words.put(constant.SYMBOL_END)
    ner_words.put(constant.SYMBOL_GO)
    ner_words.put(constant.SYMBOL_UNK)
    ner_words.put(constant.SYMBOL_QUOTE)
    for line in open(ORIG_VOCAB):
        word = line.split('\t')[0].strip()
        for prefix in constant.NER_SET:
            if re.match(r'^%s+\d+$' % prefix, word):
                ner_words.put(word)
                break
    print('NER word cnt:%s' % ner_words.qsize())

    nwords = []
    for line in open(BERT_VOCAB):
        word = line.strip()
        if word.startswith('[') and word.endswith(']') and ner_words.qsize() > 0:
            nwords.append(ner_words.get())
        else:
            nwords.append(word)
    open(OUTPUT_VOCAB, 'w').write('\n'.join(nwords))


if __name__ == '__main__':
    prepare_vocab()
