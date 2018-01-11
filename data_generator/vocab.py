from util import constant
from util.data.text_encoder import SubwordTextEncoder
from util.vocab_util import is_numeric, data_parse


class Vocab:
    def __init__(self, model_config, vocab_path=None, lower=False):
        self.model_config = model_config
        self.vocab_path = vocab_path
        if self.model_config.subword_vocab_size <= 0:
            self.init_vocab()
            if vocab_path is not None:
                self.populate_vocab()
        else:
            if vocab_path is not None:
                self.populate_subword_vocab()

    def populate_subword_vocab(self):
        self.subword = SubwordTextEncoder(self.vocab_path)
        print('Subword Vocab Populated with size %d for path %s.'
              % (len(self.subword._all_subtoken_strings), self.vocab_path))

    def init_vocab(self):
        self.w2i = {}
        self.i2w = []
        self.w2i[constant.SYMBOL_GO] = 0
        self.i2w.append(constant.SYMBOL_GO)
        self.w2i[constant.SYMBOL_PAD] = 1
        self.i2w.append(constant.SYMBOL_PAD)
        self.w2i[constant.SYMBOL_UNK] = 2
        self.i2w.append(constant.SYMBOL_UNK)
        self.w2i[constant.SYMBOL_START] = 3
        self.i2w.append(constant.SYMBOL_START)
        self.w2i[constant.SYMBOL_END] = 4
        self.i2w.append(constant.SYMBOL_END)

        for i in range(len(self.i2w), constant.REVERED_VOCAB_SIZE):
            reserved_vocab = 'REVERED_%i' % i
            self.w2i[reserved_vocab] = i
            self.i2w.append(reserved_vocab)


    def populate_vocab(self, mincount=-1):
        mincount = max(mincount, self.model_config.min_count)

        for line in open(self.vocab_path, encoding='utf-8'):
            items = line.strip().split('\t')
            w = items[0]
            if len(items) > 1:
                cnt = int(items[1])
            # else:
            #     # Accept all words
            #     cnt = 99999
            if cnt >= mincount:
                self.w2i[w] = len(self.i2w)
                self.i2w.append(w)

        print('Vocab Populated with size %d including %d reserved vocab for path %s.'
              % (len(self.i2w), constant.REVERED_VOCAB_SIZE, self.vocab_path))

    def encode(self, w):
        if self.model_config.subword_vocab_size <= 0:
            if w in self.w2i:
                return self.w2i[w]
            else:
                return self.w2i[constant.SYMBOL_UNK]
        else:
            return self.subword.encode(w)

    def contain(self, w):
        return w in self.w2i

    def describe(self, i):
        if self.model_config.subword_vocab_size <= 0:
            if i < len(self.i2w):
                return self.i2w[i]
        else:
            # Note in subword case, i should be list of id, i.e. ids.
            return self.subword.decode(i)

    def vocab_size(self):
        if self.model_config.subword_vocab_size <= 0:
            return len(self.i2w)
        else:
            return len(self.subword._all_subtoken_strings)

    @staticmethod
    def process_word(word, model_config):
        if word:
            # All numeric will map to #
            if is_numeric(word):
                return constant.SYMBOL_NUM
            else:
                if model_config.lower_case:
                    word = word.lower()
                word = data_parse(word)
                return word