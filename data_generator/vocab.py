from data_generator.vocab_config import DefaultConfig
from util import constant


class Vocab:
    def __init__(self, vocab_path=None, voc_config=None):
        self.voc_config = (DefaultConfig()
                           if voc_config is None else voc_config)
        self.vocab_path = vocab_path
        self.init_vocab()
        if vocab_path is not None:
            self.populate_vocab()

    def init_vocab(self):
        self.w2i = {}
        self.i2w = []
        self.w2i[constant.SYMBOL_PAD] = 0
        self.i2w.append(constant.SYMBOL_PAD)
        self.w2i[constant.SYMBOL_UNK] = 1
        self.i2w.append(constant.SYMBOL_UNK)
        self.w2i[constant.SYMBOL_START] = 2
        self.i2w.append(constant.SYMBOL_START)
        self.w2i[constant.SYMBOL_END] = 3
        self.i2w.append(constant.SYMBOL_END)
        self.w2i[constant.SYMBOL_GO] = 4
        self.i2w.append(constant.SYMBOL_GO)

    def populate_vocab(self, mincount=-1):
        mincount = max(mincount, self.voc_config.min_count)
        for i in range(len(self.i2w), constant.REVERED_VOCAB_SIZE):
            reserved_vocab = 'REVERED_%i' % i
            self.w2i[reserved_vocab] = i
            self.i2w.append(reserved_vocab)

        for line in open(self.vocab_path):
            items = line.strip().split('\t')
            w = items[0]
            cnt = int(items[1])
            if cnt >= mincount:
                self.w2i[w] = len(self.i2w)
                self.i2w.append(w)

        print('Vocab Populated with size %d including %d reserved vocab for path %s.'
              % (len(self.i2w), constant.REVERED_VOCAB_SIZE, self.vocab_path))


    def encode(self, w):
        if w in self.w2i:
            return self.w2i[w]
        else:
            return self.w2i[constant.SYMBOL_UNK]

    def contain(self, w):
        return w in self.w2i

    def describe(self, i):
        if i < len(self.i2w):
            return self.i2w[i]
        else:
            return constant.SYMBOL_UNK

    @staticmethod
    def process_word(word, voc_config=None):
        voc_config = (DefaultConfig()
                      if voc_config is None else voc_config)

        if word:
            # All numeric will map to #
            if word[0].isnumeric() or word[0] == '+' or word[0] == '-':
                return '#'
            # Keep mark
            elif len(word) == 1 and not word[0].isalpha():
                return word
            # Actual word
            else:
                if voc_config.lower_case:
                    word = word.lower()
                return word