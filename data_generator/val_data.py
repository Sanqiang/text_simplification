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
        self.data = self.populate_data(self.vocab_complex, self.vocab_simple, True)
        self.data_complex_raw_lines = self.populate_data_rawfile(
            self.model_config.val_dataset_complex_rawlines_file)
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
            while len(self.mapper) < len(self.data):
                self.mapper.append({})

        assert len(self.data_complex_raw_lines) == len(self.data)
        assert len(self.mapper) == len(self.data)
        for i in range(self.model_config.num_refs):
            assert len(self.data_references_raw_lines[i]) == len(self.data)
        print('Use Val Dataset: \n Simple\t %s. \n Complex\t %s. \n Size\t %d'
              % (self.model_config.val_dataset_simple_folder + self.model_config.val_dataset_simple_file,
                 self.model_config.val_dataset_complex, len(self.data)))

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

    def process_line(self, line, vocab, max_len, need_raw=False):
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
        else:
            words_raw = None

        if self.model_config.subword_vocab_size > 0:
            words = [constant.SYMBOL_START] + words + [constant.SYMBOL_END]
            words = vocab.encode(' '.join(words))
        else:
            words = [vocab.encode(word) for word in words]
            words = ([self.vocab_simple.encode(constant.SYMBOL_START)] + words +
                     [self.vocab_simple.encode(constant.SYMBOL_END)])

        if self.model_config.subword_vocab_size > 0:
            pad_id = vocab.encode(constant.SYMBOL_PAD)
        else:
            pad_id = [vocab.encode(constant.SYMBOL_PAD)]

        if len(words) < max_len:
            num_pad = max_len - len(words)
            words.extend(num_pad * pad_id)
        else:
            words = words[:max_len]

        return words, words_raw

    def populate_data(self, vocab_comp, vocab_simp, need_raw=False):
        # Populate data into memory
        data = []
        # max_len = -1
        # from collections import Counter
        # len_report = Counter()
        lines_comp = open(
            self.model_config.val_dataset_complex, encoding='utf-8').readlines()
        lines_simp = open(
            self.model_config.val_dataset_simple_folder + self.model_config.val_dataset_simple_file,
            encoding='utf-8').readlines()
        assert len(lines_comp) == len(lines_simp)
        for line_id in range(len(lines_comp)):
            obj = {}
            line_comp = lines_comp[line_id]
            line_simp = lines_simp[line_id]
            words_comp, words_raw_comp = self.process_line(
                line_comp, vocab_comp, self.model_config.max_complex_sentence, need_raw)
            words_simp, words_raw_simp = self.process_line(
                line_simp, vocab_simp, self.model_config.max_simple_sentence, need_raw)

            obj['words_comp'] = words_comp
            obj['words_simp'] = words_simp
            if need_raw:
                obj['words_raw_comp'] = words_raw_comp
                obj['words_raw_simp'] = words_raw_simp

            oov = {}
            if self.model_config.pointer_mode == 'ptr':
                oov['w2i'] = {}
                oov['i2w'] = []

                for idx, wid in enumerate(words_comp):
                    if wid == vocab_comp.encode(constant.SYMBOL_UNK):
                        word_raw = words_raw_comp[idx]
                        if word_raw not in oov['w2i']:
                            oov['w2i'][word_raw] = len(oov['i2w'])
                            oov['i2w'].append(word_raw)

                # for idx, wid in enumerate(words_simp):
                #     if wid == vocab_simp.encode(constant.SYMBOL_UNK):
                #         word_raw = words_raw_simp[idx]
                #         if word_raw not in oov['w2i']:
                #             oov['w2i'][word_raw] = len(oov['i2w'])
                #             oov['i2w'].append(word_raw)
            obj['oov'] = oov

            data.append(obj)
            # len_report.update([len(words)])
            # if len(words) > max_len:
            #     max_len = len(words)
        # print('Max length for data %s is %s.' % (data_path, max_len))
        # print('counter:%s' % len_report)
        return data

    def get_data_iter(self):
        i = 0
        while True:
            ref_rawlines_batch = [self.data_references_raw_lines[j][i]
                                  for j in range(self.model_config.num_refs)]
            supplement = {}
            if self.model_config.memory == 'rule':
                supplement['mem'] = self.rules[i]

            obj = {
                'sentence_simple': self.data[i]['words_simp'],
                'sentence_complex': self.data[i]['words_comp'],
                'sentence_complex_raw': self.data[i]['words_raw_comp'],
                'sentence_simple_raw': self.data[i]['words_raw_simp'],
                'sentence_complex_raw_lines': self.data_complex_raw_lines[i],
                'mapper': self.mapper[i],
                'ref_raw_lines': ref_rawlines_batch,
                'sup': supplement,
                'oov': self.data[i]['oov']
            }

            yield obj

            i += 1
            if i == len(self.data):
                yield None
