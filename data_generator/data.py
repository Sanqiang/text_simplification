import random as rd
import copy as cp

from data_generator.vocab import Vocab
from nltk import word_tokenize
from util import constant

class Data:
    def __init__(self, data_simple_path, data_complex_path, vocab_simple_path, vocab_complex_path):
        self.vocab_simple = Vocab(vocab_simple_path)
        self.vocab_complex = Vocab(vocab_complex_path)
        self.data_simple = self.populate_data(data_simple_path, self.vocab_simple)
        self.data_complex = self.populate_data(data_complex_path, self.vocab_complex)
        self.size = len(self.data_simple)

    def populate_data(self, data_path, vocab):
        # Populate data into memory
        data = []
        for line in open(data_path):
            words = word_tokenize(line)
            words = [Vocab.process_word(word)
                     for word in words]
            words = [vocab.encode(word) for word in words]
            words = ([self.vocab_simple.encode(constant.SYMBOL_START)] + words +
                     [self.vocab_simple.encode(constant.SYMBOL_END)])

            data.append(words)
        return data

    def get_data_sample(self):
        i = rd.sample(range(self.size), 1)[0]
        return cp.deepcopy(self.data_simple[i]), cp.deepcopy(self.data_complex[i])

