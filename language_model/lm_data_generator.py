from data_generator.vocab import Vocab
from util import constant

from types import SimpleNamespace
from os import listdir
import random as rd

from language_model.lm_arguments import get_args

args = get_args()


class LM_Data:
    def __init__(self):
        self.model_config = SimpleNamespace(min_count=args.min_count, subword_vocab_size=0, lower_case=True)
        self.vocab = Vocab(self.model_config, args.vocab_path)

        files = listdir(args.data_path)
        self.data_paths = []
        for file in files:
            data_path = args.data_path + file
            f = open(data_path)
            nsamples= 0
            for _ in f:
                nsamples += 1
            f.close()
            self.data_paths.append((data_path, nsamples))
            print('Add data path %s with count %s' % (data_path, nsamples))

    def get_data_sample_it(self):
        self.cur_data_path, self.cur_samples = rd.choice(self.data_paths)
        f = open(self.cur_data_path, encoding='utf-8')
        i = 0
        while True:
            if i == self.cur_samples-1:
                self.cur_data_path, self.cur_samples = rd.choice(self.data_paths)
                f = open(self.cur_data_path, encoding='utf-8')
                i = 0
            line = f.readline()
            words = line.split()
            words = [Vocab.process_word(word, self.model_config).lower()
                     for word in words]
            words = [self.vocab.encode(word) for word in words]
            words = ([self.vocab.encode(constant.SYMBOL_START)] + words +
                     [self.vocab.encode(constant.SYMBOL_END)])
            yield words
            i += 1