from data_generator.vocab import Vocab
from nltk import word_tokenize
from util import constant
from util.map_util import load_mappers
from data_generator.rule import Rule
from data_generator import data_utils


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
            self.model_config.val_dataset_complex_rawlines_file,
            self.model_config.lower_case)
        # Populate simple references
        self.data_references_raw_lines = []
        for i in range(self.model_config.num_refs):
            ref_tmp_rawlines = self.populate_data_rawfile(
                self.model_config.val_dataset_simple_folder +
                self.model_config.val_dataset_simple_rawlines_file_references +
                str(i), self.model_config.lower_case)
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

        if 'rule' in self.model_config.memory or 'direct' in self.model_config.memory:
            self.vocab_rule = Rule(model_config, self.model_config.vocab_rules)
            self.rules = self.populate_rules(
                self.model_config.val_dataset_complex_ppdb, self.vocab_rule)
            print('Populate Rule with size:%s' % self.vocab_rule.get_rule_size())

        if self.model_config.tune_style:
            self.comp_features = self.populate_comp_features(
                self.model_config.val_dataset_complex_features)

    def populate_comp_features(self, feature_path):
        data = []
        for line in open(feature_path, encoding='utf-8'):
            items = line.split('\t')
            data.append(
                (float(items[0]), float(items[1])))
        return data

    def populate_rules(self, rule_path, vocab_rule):
        data = []
        for line in open(rule_path, encoding='utf-8'):
            cur_rules = line.split('\t')
            tmp = []
            for cur_rule in cur_rules:
                rule_id, _, rule_targets = vocab_rule.encode(cur_rule)
                if rule_targets is not None:
                    if self.model_config.subword_vocab_size or 'bert_token' in self.model_config.bert_mode:
                        tmp.append((rule_id, self.vocab_simple.encode(rule_targets)))
                    else:
                        tmp.append((rule_id, [self.vocab_simple.encode(rule_target) for rule_target in rule_targets]))
            data.append(tmp)
        return data

    def populate_data_rawfile(self, data_path, lower_case=True):
        """Populate data raw lines into memory"""
        data = []
        for line in open(data_path, encoding='utf-8'):
            if lower_case:
                line = line.lower()
            data.append(line.strip())
        return data

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
            words_comp, words_raw_comp, obj_comp = data_utils.process_line(
                line_comp, vocab_comp, self.model_config.max_complex_sentence, self.model_config, need_raw,
                self.model_config.lower_case)
            words_simp, words_raw_simp, obj_simp = data_utils.process_line(
                line_simp, vocab_simp, self.model_config.max_simple_sentence, self.model_config, need_raw,
                self.model_config.lower_case)

            obj['words_comp'] = words_comp
            obj['words_simp'] = words_simp
            if need_raw:
                obj['words_raw_comp'] = words_raw_comp
                obj['words_raw_simp'] = words_raw_simp
            if self.model_config.subword_vocab_size and self.model_config.seg_mode:
                obj['line_comp_segids'] = obj_comp['segment_idxs']
                obj['line_simp_segids'] = obj_simp['segment_idxs']

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
            if i >= len(self.data):
                yield None
            else:
                ref_rawlines_batch = [self.data_references_raw_lines[j][i]
                                      for j in range(self.model_config.num_refs)]
                supplement = {}
                if 'rule' in self.model_config.memory or 'direct' in self.model_config.memory:
                    supplement['mem'] = self.rules[i]

                if self.model_config.tune_style:
                    supplement['comp_features'] = self.comp_features[i]

                obj = {
                    'sentence_simple': self.data[i]['words_simp'],
                    'sentence_complex': self.data[i]['words_comp'],
                    'sentence_complex_raw': self.data[i]['words_raw_comp'],
                    'sentence_simple_raw': self.data[i]['words_raw_simp'],
                    'sentence_complex_raw_lines': self.data_complex_raw_lines[i],
                    'mapper': self.mapper[i],
                    'ref_raw_lines': ref_rawlines_batch,
                    'sup': supplement,
                }

                if self.model_config.subword_vocab_size and self.model_config.seg_mode:
                    obj['line_comp_segids'] = self.data[i]['line_comp_segids']
                    obj['line_simp_segids'] = self.data[i]['line_simp_segids']

                yield obj

                i += 1
