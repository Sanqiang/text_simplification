import random as rd
import copy as cp

from data_generator.vocab import Vocab
from nltk import word_tokenize
from util import constant


class TrainData:
    def __init__(self, model_config):
        self.model_config = model_config
        vocab_simple_path = self.model_config.vocab_simple
        vocab_complex_path = self.model_config.vocab_complex
        vocab_all_path = self.model_config.vocab_all
        data_simple_path = self.model_config.train_dataset_simple
        data_complex_path = self.model_config.train_dataset_complex

        if (self.model_config.tie_embedding == 'none' or
                    self.model_config.tie_embedding == 'dec_out'):
            self.vocab_simple = Vocab(model_config, vocab_simple_path)
            self.vocab_complex = Vocab(model_config, vocab_complex_path)
        elif (self.model_config.tie_embedding == 'all' or
                    self.model_config.tie_embedding == 'enc_dec'):
            self.vocab_simple = Vocab(model_config, vocab_all_path)
            self.vocab_complex = Vocab(model_config, vocab_all_path)

        # Populate basic complex simple pairs
        self.data_simple = self.populate_data(data_simple_path, self.vocab_simple)
        self.data_complex = self.populate_data(data_complex_path, self.vocab_complex)
        self.size = len(self.data_simple)
        print('Use Train Dataset: \n Simple\t %s. \n Complex\t %s. \n Size\t %d'
              % (data_simple_path, data_complex_path, self.size))

    def populate_data(self, data_path, vocab):
        # Populate data into memory
        data = []
        for line in open(data_path, encoding='utf-8'):
            # line = line.split('\t')[2]
            if self.model_config.tokenizer == 'split':
                words = line.split()
            elif self.model_config.tokenizer == 'nltk':
                words = word_tokenize(line)
            else:
                raise Exception('Unknown tokenizer.')

            words = [Vocab.process_word(word, self.model_config)
                     for word in words]
            words = [vocab.encode(word) for word in words]
            words = ([self.vocab_simple.encode(constant.SYMBOL_START)] + words +
                     [self.vocab_simple.encode(constant.SYMBOL_END)])

            data.append(words)
        return data

    def get_data_sample(self):
        i = rd.sample(range(self.size), 1)[0]
        return cp.deepcopy(self.data_simple[i]), cp.deepcopy(self.data_complex[i])

    def get_data_iter(self):
        i = 0
        while True:
            yield cp.deepcopy(self.data_simple[i]), cp.deepcopy(self.data_complex[i])
            i += 1
            if i == len(self.data_simple):
                yield None, None

