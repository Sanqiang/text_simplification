from data_generator.vocab import Vocab
from nltk import word_tokenize
from util import constant
from util.map_util import load_mappers
from model.ppdb import PPDB
from data_generator.rule import Rule

import copy as cp


class ValData:
    def __init__(self, model_config):
        self.model_config = model_config

        vocab_simple_path = self.model_config.vocab_simple
        vocab_complex_path = self.model_config.vocab_complex
        vocab_all_path = self.model_config.vocab_all
        if self.model_config.subword_vocab_size > 0:
            vocab_simple_path = self.model_config.subword_vocab_simple
            vocab_complex_path = self.model_config.subword_vocab_complex
            vocab_all_path = self.model_config.subword_vocab_all

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
        self.data_complex_raw_lines = self.populate_data_rawfile(
            self.model_config.val_dataset_complex_rawlines_file)
        self.data_simple, _ = self.populate_data(
            self.model_config.val_dataset_simple_folder + self.model_config.val_dataset_simple_file,
            self.vocab_simple)
        # Populate simple references
        self.data_references_raw_lines = []
        for i in range(self.model_config.num_refs):
            ref_tmp_rawlines = self.populate_data_rawfile(
                self.model_config.val_dataset_simple_folder +
                self.model_config.val_dataset_simple_rawlines_file_references +
                str(i))
            self.data_references_raw_lines.append(ref_tmp_rawlines)

        if self.model_config.replace_ner:
            self.mapper = load_mappers(self.model_config.val_mapper, self.model_config.lower_case)
            while len(self.mapper) < len(self.data_simple):
                self.mapper.append({})

        self.size = len(self.data_simple)
        assert len(self.data_complex) == self.size
        assert len(self.data_complex_raw) == self.size
        assert len(self.data_complex_raw_lines) == self.size
        assert len(self.mapper) == self.size
        for i in range(self.model_config.num_refs):
            assert len(self.data_references_raw_lines[i]) == self.size
        print('Use Val Dataset: \n Simple\t %s. \n Complex\t %s. \n Size\t %d'
              % (self.model_config.val_dataset_simple_folder + self.model_config.val_dataset_simple_file,
                 self.model_config.val_dataset_complex, self.size))

        if model_config.ppdb_emode != 'none':
            self.ppdb = PPDB(model_config)

        if self.model_config.memory == 'rule':
            self.vocab_rule = Rule(model_config, self.model_config.vocab_rules)
            self.rules = self.populate_rules(
                self.model_config.val_dataset_complex_ppdb, self.vocab_rule)
            assert len(self.rules) == self.size

    def populate_rules(self, rule_path, vocab_rule):
        data = []
        for line in open(rule_path, encoding='utf-8'):
            cur_rules = line.split('\t')
            tmp = []
            for cur_rule in cur_rules:
                rule_id, rule_targets = vocab_rule.encode(cur_rule)
                if rule_targets is not None:
                    tmp.append((rule_id, [self.vocab_simple.encode(rule_target) for rule_target in rule_targets]))
            data.append(tmp)
        return data

    def populate_data_rawfile(self, data_path):
        """Populate data raw lines into memory"""
        data = []
        for line in open(data_path, encoding='utf-8'):
            data.append(line.strip())
        return data

    def populate_data(self, data_path, vocab, need_raw=False):
        # Populate data into memory
        data = []
        data_raw = []
        # max_len = -1
        # from collections import Counter
        # len_report = Counter()
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

            if self.model_config.subword_vocab_size > 0:
                words = [constant.SYMBOL_START] + words + [constant.SYMBOL_END]
                words = vocab.encode(' '.join(words))
            else:
                words = [vocab.encode(word) for word in words]
                words = ([self.vocab_simple.encode(constant.SYMBOL_START)] + words +
                         [self.vocab_simple.encode(constant.SYMBOL_END)])

            data.append(words)
            # len_report.update([len(words)])
            # if len(words) > max_len:
            #     max_len = len(words)
        # print('Max length for data %s is %s.' % (data_path, max_len))
        # print('counter:%s' % len_report)
        return data, data_raw

    def get_data_iter(self):
        i = 0
        while True:
            ref_rawlines_batch = cp.deepcopy([self.data_references_raw_lines[j][i]
                                         for j in range(self.model_config.num_refs)])
            supplement = {}
            if self.model_config.memory == 'rule':
                supplement['mem'] = self.rules[i]

            yield (cp.deepcopy(self.data_simple[i]),
                   cp.deepcopy(self.data_complex[i]),
                   cp.deepcopy(self.data_complex_raw[i]),
                   cp.deepcopy(self.data_complex_raw_lines[i]),
                   self.mapper[i],
                   ref_rawlines_batch,
                   supplement)

            i += 1
            if i == len(self.data_simple):
                yield None, None, None, None, None, None, None
