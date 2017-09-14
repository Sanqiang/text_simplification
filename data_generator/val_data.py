from data_generator.vocab import Vocab
from nltk import word_tokenize
from util import constant

import copy as cp


class ValData:
    def __init__(self, vocab_simple_path, vocab_complex_path, model_config):
        self.vocab_simple = Vocab(vocab_simple_path)
        self.vocab_complex = Vocab(vocab_complex_path)
        # Populate basic complex simple pairs
        self.data_complex = self.populate_data(model_config.val_dataset_complex, self.vocab_simple)
        self.data_simple = self.populate_data(
            model_config.val_dataset_simple_folder + model_config.val_dataset_simple_file,
            self.vocab_complex)
        # Populate simple references
        self.data_references = []
        for i in range(model_config.num_refs):
            self.data_references.append(
                self.populate_data(model_config.val_dataset_simple_folder +
                                   model_config.val_dataset_simple_references +
                                   str(i), self.vocab_simple))

        self.size = len(self.data_simple)

    def populate_data(self, data_path, vocab):
        # Populate data into memory
        data = []
        for line in open(data_path, encoding='utf-8'):
            words = word_tokenize(line)
            words = [Vocab.process_word(word)
                     for word in words]
            words = [vocab.encode(word) for word in words]
            words = ([self.vocab_simple.encode(constant.SYMBOL_START)] + words +
                     [self.vocab_simple.encode(constant.SYMBOL_END)])

            data.append(words)
        return data

    def get_data_iter(self):
        i = 0
        while True:
            yield (cp.deepcopy(self.data_simple[i]), cp.deepcopy(self.data_complex[i]),
                   (cp.deepcopy(self.data_references[0][i]), cp.deepcopy(self.data_references[1][i]),
                    cp.deepcopy(self.data_references[2][i]), cp.deepcopy(self.data_references[3][i]),
                    cp.deepcopy(self.data_references[4][i]), cp.deepcopy(self.data_references[5][i]),
                    cp.deepcopy(self.data_references[6][i]), cp.deepcopy(self.data_references[7][i])))
            if i == len(self.data):
                return None