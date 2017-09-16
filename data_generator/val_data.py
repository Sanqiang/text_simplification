from data_generator.vocab import Vocab
from nltk import word_tokenize
from util import constant

import copy as cp


class ValData:
    def __init__(self, model_config, vocab_simple_path, vocab_complex_path):
        self.model_config = model_config
        self.vocab_simple = Vocab(model_config, vocab_simple_path)
        self.vocab_complex = Vocab(model_config, vocab_complex_path)
        # Populate basic complex simple pairs
        self.data_complex = self.populate_data(self.model_config.val_dataset_complex, self.vocab_complex)
        self.data_simple = self.populate_data(
            self.model_config.val_dataset_simple_folder + self.model_config.val_dataset_simple_file,
            self.vocab_simple)
        # Populate simple references
        self.data_references = []
        for i in range(self.model_config.num_refs):
            self.data_references.append(
                self.populate_data(self.model_config.val_dataset_simple_folder +
                                   self.model_config.val_dataset_simple_references +
                                   str(i), self.vocab_simple))

        self.size = len(self.data_simple)
        print('Use Val Dataset: \n Simple\t %s. \n Complex\t %s. \n Size\t %d'
              % (self.model_config.val_dataset_simple_folder + self.model_config.val_dataset_simple_file,
                 self.model_config.val_dataset_complex, self.size))

    def populate_data(self, data_path, vocab):
        # Populate data into memory
        data = []
        for line in open(data_path):
            words = word_tokenize(line)
            words = [Vocab.process_word(word, self.model_config)
                     for word in words]
            words = [vocab.encode(word) for word in words]
            words = ([self.vocab_simple.encode(constant.SYMBOL_START)] + words +
                     [self.vocab_simple.encode(constant.SYMBOL_END)])

            data.append(words)
        return data

    def get_data_iter(self):
        i = 0
        while True:
            if self.model_config.num_refs == 8:
                yield cp.deepcopy(self.data_simple[i]), cp.deepcopy(self.data_complex[i]),\
                      (cp.deepcopy(self.data_references[0][i]), cp.deepcopy(self.data_references[1][i]),
                       cp.deepcopy(self.data_references[2][i]), cp.deepcopy(self.data_references[3][i]),
                       cp.deepcopy(self.data_references[4][i]), cp.deepcopy(self.data_references[5][i]),
                       cp.deepcopy(self.data_references[6][i]), cp.deepcopy(self.data_references[7][i]))
            elif self.model_config.num_refs == 0:
                yield cp.deepcopy(self.data_simple[i]), cp.deepcopy(self.data_complex[i]), None
            i += 1
            if i == len(self.data_simple):
                yield None, None, None
