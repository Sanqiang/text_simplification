from data_generator.vocab import Vocab
from nltk import word_tokenize
from util import constant
from util.map_util import load_mappers

import copy as cp


class ValData:
    def __init__(self, model_config, ):
        self.model_config = model_config
        vocab_simple_path = self.model_config.vocab_simple
        vocab_complex_path = self.model_config.vocab_complex
        vocab_all_path = self.model_config.vocab_all

        if (self.model_config.tie_embedding == 'none' or
                    self.model_config.tie_embedding == 'dec_out'):
            self.vocab_simple = Vocab(model_config, vocab_simple_path)
            self.vocab_complex = Vocab(model_config, vocab_complex_path)
        elif (self.model_config.tie_embedding == 'all' or
                    self.model_config.tie_embedding == 'enc_dec'):
            self.vocab_simple = Vocab(model_config, vocab_all_path)
            self.vocab_complex = Vocab(model_config, vocab_all_path)

        # Populate basic complex simple pairs
        self.data_complex, self.data_complex_raw = self.populate_data(
            self.model_config.val_dataset_complex, self.vocab_complex, need_raw=True)
        self.data_simple, _ = self.populate_data(
            self.model_config.val_dataset_simple_folder + self.model_config.val_dataset_simple_file,
            self.vocab_simple)
        # Populate simple references
        self.data_references = []
        for i in range(self.model_config.num_refs):
            self.data_references.append(
                self.populate_data(self.model_config.val_dataset_simple_folder +
                                   self.model_config.val_dataset_simple_references +
                                   str(i), self.vocab_simple)[0])

        if self.model_config.replace_ner:
            self.mapper = load_mappers(self.model_config.val_mapper, self.model_config.lower_case)
            while len(self.mapper) < len(len(self.data_simple)):
                self.mapper.append({})

        self.size = len(self.data_simple)
        assert len(self.data_complex) == self.size
        assert len(self.data_complex_raw) == self.size
        assert len(self.mapper) == self.size
        for i in range(self.model_config.num_refs):
            assert len(self.data_references[i]) == self.size
        print('Use Val Dataset: \n Simple\t %s. \n Complex\t %s. \n Size\t %d'
              % (self.model_config.val_dataset_simple_folder + self.model_config.val_dataset_simple_file,
                 self.model_config.val_dataset_complex, self.size))

    def populate_data(self, data_path, vocab, need_raw=False):
        # Populate data into memory
        data = []
        data_raw = []
        for line in open(data_path, encoding='utf-8'):
            if self.model_config.tokenizer == 'split':
                words = line.split()
            elif self.model_config.tokenizer == 'nltk':
                words = word_tokenize(line)
            else:
                raise Exception('Unknown tokenizer.')
            words = [Vocab.process_word(word, self.model_config)
                     for word in words]
            if need_raw:
                words_raw = [constant.SYMBOL_START] + words + [constant.SYMBOL_END]
                data_raw.append(words_raw)
            words = [vocab.encode(word) for word in words]
            words = ([self.vocab_simple.encode(constant.SYMBOL_START)] + words +
                     [self.vocab_simple.encode(constant.SYMBOL_END)])

            data.append(words)
        return data, data_raw

    def get_data_iter(self):
        i = 0
        while True:
            ref_batch = cp.deepcopy([self.data_references[j][i] for j in range(self.model_config.num_refs)])
            yield (cp.deepcopy(self.data_simple[i]), cp.deepcopy(self.data_complex[i]),
                   cp.deepcopy(self.data_complex_raw[i]), self.mapper[i], ref_batch)

            i += 1
            if i == len(self.data_simple):
                yield None, None, None, None, None
