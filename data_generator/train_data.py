import copy as cp
import random as rd
import tensorflow as tf
import glob

import numpy as np
from nltk import word_tokenize
from copy import deepcopy
import time
import random as rd

from data_generator.vocab import Vocab
from data_generator.rule import Rule
from util import constant
from data_generator import data_utils


# Deprecated: use Tf.Example (TfExampleTrainDataset) instead
class TrainData:
    """Fetching training dataset from plain data."""

    def __init__(self, model_config):
        self.model_config = model_config

        vocab_simple_path = self.model_config.vocab_simple
        vocab_complex_path = self.model_config.vocab_complex
        vocab_all_path = self.model_config.vocab_all
        if self.model_config.subword_vocab_size > 0:
            vocab_simple_path = self.model_config.subword_vocab_simple
            vocab_complex_path = self.model_config.subword_vocab_complex
            vocab_all_path = self.model_config.subword_vocab_all

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

        self.size = self.get_size(data_complex_path)
        # Populate basic complex simple pairs
        if not self.model_config.it_train:
            self.data = self.populate_data(data_complex_path, data_simple_path,
                                           self.vocab_complex, self.vocab_simple, True)
        else:
            self.data_it = self.get_data_sample_it(data_simple_path, data_complex_path)

        print('Use Train Dataset: \n Simple\t %s. \n Complex\t %s. \n Size\t %d.'
              % (data_simple_path, data_complex_path, self.size))

        if 'rule' in self.model_config.memory or 'rule' in self.model_config.rl_configs:
            self.vocab_rule = Rule(model_config, self.model_config.vocab_rules)
            self.rules_target = self.populate_rules(
                self.model_config.train_dataset_complex_ppdb, self.vocab_rule)
            assert len(self.rules_target) == self.size
            print('Populate Rule with size:%s' % self.vocab_rule.get_rule_size())
        if model_config.pretrained:
            self.init_pretrained_embedding()

    def get_size(self, data_complex_path):
        return len(open(data_complex_path, encoding='utf-8').readlines())

    def get_data_sample_it(self, data_simple_path, data_complex_path):
        f_simple = open(data_simple_path, encoding='utf-8')
        f_complex = open(data_complex_path, encoding='utf-8')
        i = 0
        while True:
            if i >= self.size:
                f_simple = open(data_simple_path, encoding='utf-8')
                f_complex = open(data_complex_path, encoding='utf-8')
                i = 0
            line_complex = f_complex.readline()
            line_simple = f_simple.readline()
            if rd.random() < 0.5 or i >= self.size:
                i += 1
                continue

            words_complex, words_raw_comp, _ = data_utils.process_line(
                line_complex, self.vocab_complex, self.model_config.max_complex_sentence, self.model_config, True)
            words_simple, words_raw_simp, _ = data_utils.process_line(
                line_simple, self.vocab_simple, self.model_config.max_simple_sentence, self.model_config, True)

            supplement = {}
            if 'rule' in self.model_config.memory or 'direct' in self.model_config.memory:
                supplement['rules_target'] = self.rules_target[i]

            obj = {}
            obj['words_comp'] = words_complex
            obj['words_simp'] = words_simple
            obj['words_raw_comp'] = words_raw_comp
            obj['words_raw_simp'] = words_raw_simp

            yield i, obj, supplement

            i += 1

    def populate_rules(self, rule_path, vocab_rule):
        data_target = []
        for line in open(rule_path, encoding='utf-8'):
            cur_rules = line.split('\t')
            tmp, tmp_align = [], []
            for cur_rule in cur_rules:
                rule_id, rule_origins, rule_targets = vocab_rule.encode(cur_rule)
                if rule_targets is not None and rule_origins is not None:
                    tmp.append((rule_id, rule_targets))
            data_target.append(tmp)

        return data_target

    def populate_data(self, data_path_comp, data_path_simp, vocab_comp, vocab_simp, need_raw=False):
        # Populate data into memory
        data = []
        # max_len = -1
        # from collections import Counter
        # len_report = Counter()
        lines_comp = open(data_path_comp, encoding='utf-8').readlines()
        lines_simp = open(data_path_simp, encoding='utf-8').readlines()
        assert len(lines_comp) == len(lines_simp)
        for line_id in range(len(lines_comp)):
            obj = {}
            line_comp = lines_comp[line_id]
            line_simp = lines_simp[line_id]
            words_comp, words_raw_comp, _ = data_utils.process_line(
                line_comp, vocab_comp, self.model_config.max_complex_sentence, self.model_config, need_raw)
            words_simp, words_raw_simp, _ = data_utils.process_line(
                line_simp, vocab_simp, self.model_config.max_simple_sentence, self.model_config, need_raw)
            obj['words_comp'] = words_comp
            obj['words_simp'] = words_simp
            if need_raw:
                obj['words_raw_comp'] = words_raw_comp
                obj['words_raw_simp'] = words_raw_simp

            data.append(obj)
        return data

    def get_data_sample(self):
        i = rd.sample(range(self.size), 1)[0]
        supplement = {}
        if 'rule' in self.model_config.memory or 'direct' in self.model_config.memory:
            supplement['rules_target'] = self.rules_target[i]
            supplement['rules_align'] = self.rules_align[i]

        return i, self.data[i], supplement

    def init_pretrained_embedding(self):
        if self.model_config.subword_vocab_size > 0:
            # Subword doesn't need pretrained embedding.
            return

        if self.model_config.pretrained_embedding is None:
            return

        print('Use Pretrained Embedding\t%s.' % self.model_config.pretrained_embedding)

        if not hasattr(self, 'glove'):
            self.glove = {}
            for line in open(self.model_config.pretrained_embedding, encoding='utf-8'):
                pairs = line.split()
                word = ' '.join(pairs[:-self.model_config.dimension])
                if word in self.vocab_simple.w2i or word in self.vocab_complex.w2i:
                    embedding = pairs[-self.model_config.dimension:]
                    self.glove[word] = embedding

            # For vocabulary complex
            pretrained_cnt = 0
            random_cnt = 0
            self.pretrained_emb_complex = np.empty(
                (self.vocab_complex.vocab_size(), self.model_config.dimension), dtype=np.float32)
            for wid, word in enumerate(self.vocab_complex.i2w):
                if word in self.glove:
                    n_vector = np.array(self.glove[word])

                    self.pretrained_emb_complex[wid, :] = n_vector
                    pretrained_cnt += 1
                else:
                    n_vector = np.array([np.random.uniform(-0.08, 0.08)
                                         for _ in range(self.model_config.dimension)])
                    self.pretrained_emb_complex[wid, :] = n_vector
                    random_cnt += 1
            assert self.vocab_complex.vocab_size() == random_cnt + pretrained_cnt
            print(
                'For Vocab Complex, %s words initialized with pretrained vector, '
                'other %s words initialized randomly.' %
                (pretrained_cnt, random_cnt))

            # For vocabulary simple
            pretrained_cnt = 0
            random_cnt = 0
            self.pretrained_emb_simple = np.empty(
                (len(self.vocab_simple.i2w), self.model_config.dimension), dtype=np.float32)
            for wid, word in enumerate(self.vocab_simple.i2w):
                if word in self.glove:
                    n_vector = np.array(self.glove[word])
                    self.pretrained_emb_simple[wid, :] = n_vector
                    pretrained_cnt += 1
                else:
                    n_vector = np.array([np.random.uniform(-0.08, 0.08)
                                         for _ in range(self.model_config.dimension)])
                    self.pretrained_emb_simple[wid, :] = n_vector
                    random_cnt += 1
            assert len(self.vocab_simple.i2w) == random_cnt + pretrained_cnt
            print(
                'For Vocab Simple, %s words initialized with pretrained vector, '
                'other %s words initialized randomly.' %
                (pretrained_cnt, random_cnt))

            del self.glove


class TfExampleTrainDataset():
    """Fetching training dataset from tf.example Dataset"""

    def __init__(self, model_config):
        self.model_config = model_config

        if self.model_config.subword_vocab_size:
            vocab_simple_path = self.model_config.subword_vocab_simple
            vocab_complex_path = self.model_config.subword_vocab_complex
            vocab_all_path = self.model_config.subword_vocab_all
        else:
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

        self.feature_set = {
            'line_comp': tf.FixedLenFeature([], tf.string),
            'line_simp': tf.FixedLenFeature([], tf.string),
        }
        if self.model_config.tune_style:
            self.feature_set['ppdb_score'] = tf.FixedLenFeature([], tf.float32)
            self.feature_set['len_score'] = tf.FixedLenFeature([], tf.float32)
            self.feature_set['add_score'] = tf.FixedLenFeature([], tf.float32)
            self.feature_set['dsim_score'] = tf.FixedLenFeature([], tf.float32)

        if 'rule' in self.model_config.memory or 'direct' in self.model_config.memory:
            self.feature_set['ppdb_rule'] = tf.FixedLenFeature([], tf.string)
            self.vocab_rule = Rule(model_config, self.model_config.vocab_rules)

        if self.model_config.dmode == 'listalter':
            assert type(self.model_config.train_dataset) == list
            self.dataset = []
            self.iterator = []
            self.training_init_op = []
            for dataset_path in self.model_config.train_dataset:
                dataset = self._get_dataset(glob.glob(dataset_path))
                iterator = tf.data.Iterator.from_structure(
                    dataset.output_types,
                    dataset.output_shapes)
                init_op = iterator.make_initializer(dataset)
                self.dataset.append(dataset)
                self.iterator.append(iterator)
                self.training_init_op.append(init_op)
        else:
            self.dataset = self._get_dataset(glob.glob(self.model_config.train_dataset))
            self.iterator = tf.data.Iterator.from_structure(
                self.dataset.output_types,
                self.dataset.output_shapes)
            self.training_init_op = self.iterator.make_initializer(self.dataset)

            if self.model_config.dmode == 'alter':
                self.dataset2 = self._get_dataset(glob.glob(self.model_config.train_dataset2))
                self.iterator2 = tf.data.Iterator.from_structure(
                    self.dataset2.output_types,
                    self.dataset2.output_shapes)
                self.training_init_op2 = self.iterator2.make_initializer(self.dataset2)

    def get_data_sample(self):
        if self.model_config.dmode == 'listalter':
            assert type(self.iterator) == list
            return rd.choice(self.iterator).get_next()
        elif rd.random() >= 0.5 or self.model_config.dmode != 'alter':
            return self.iterator.get_next()
        else:
            return self.iterator2.get_next()

    def _parse(self, serialized_example):
        features = tf.parse_single_example(serialized_example, features=self.feature_set)

        def process_line_pair(line_complex, line_simple):
            words_simple, _, obj_simple = data_utils.process_line(
                line_simple, self.vocab_simple, self.model_config.max_simple_sentence, self.model_config, True)
            words_complex, _, obj_complex = data_utils.process_line(
                line_complex, self.vocab_complex, self.model_config.max_complex_sentence, self.model_config, True,
                base_line=line_simple)

            if self.model_config.subword_vocab_size and self.model_config.seg_mode:

                seg_comp_idxs, seg_simp_idxs = obj_complex['segment_idxs'], obj_simple['segment_idxs']

                # seg_comp_idxs = np.array(seg_comp_idxs)
                # seg_simp_idxs = np.array(seg_simp_idxs)
                # words_simple2 = np.array([self.vocab_simple.subword.decode([w]) for w in words_simple])
                # words_complex2 = np.array([self.vocab_complex.subword.decode([w]) for w in words_complex])
                # x1 = words_complex2[np.where(seg_comp_idxs > 0)]
                # x2 = words_simple2[np.where(seg_simp_idxs > 0)]
                # if any([v>0 for v in seg_comp_idxs]) and (len(x1) != len(x2) or any(x1!=x2)):
                #     print('line_complex=%s' % line_complex)
                #     print('line_simple=%s' % line_simple)
                #     print('words_complex=%s' % words_complex)
                #     print('words_simple=%s' % words_simple)
                #     print('words_complex2=%s' % [self.vocab_complex.subword.decode([w]) for w in words_complex])
                #     print('words_simple2=%s' % [self.vocab_simple.subword.decode([w]) for w in words_simple])
                #     print('seg_comp_idxs=%s' % seg_comp_idxs)
                #     print('seg_simp_idxs=%s' % seg_simp_idxs)
                #     print(x1)
                #     print(x2)

                # if 'cp' in self.model_config.seg_mode:
                #     seg_idxs = [1 if p[0] > 0 and p[1] > 0 else 0 for p in zip(seg_comp_idxs, seg_simp_idxs)]
                #     seg_comp_idxs, seg_simp_idxs = seg_idxs, seg_idxs
                #     print('seg_idxs:%s' % seg_idxs)

                return (np.array(words_complex, np.int32), np.array(words_simple, np.int32),
                        np.array(seg_comp_idxs, np.int32), np.array(seg_simp_idxs, np.int32))
            else:
                return np.array(words_complex, np.int32), np.array(words_simple, np.int32)

        def process_rule(ppdb_rule):
            rule_ids, rule_tars = [], []
            rules = ppdb_rule.decode("utf-8") .split('\t')
            for rule in rules:
                rule_id, rule_origin, rule_target = self.vocab_rule.encode(rule)
                if rule_target is not None and rule_origin is not None:
                    rule_ids.append(rule_id)
                    if 'rule' in self.model_config.memory:
                        rule_tars.append(rule_target)
                    elif 'direct' in self.model_config.memory:
                        rule_tars.extend(self.vocab_simple.encode(rule_target))

                if len(rule_ids) < self.model_config.max_cand_rules:
                    num_pad = self.model_config.max_cand_rules - len(rule_ids)
                    rule_ids.extend(num_pad * [0])
                    if 'rule' in self.model_config.memory:
                        rule_tars.extend(num_pad * [constant.SYMBOL_PAD])
                    elif 'direct' in self.model_config.memory:
                        num_pad = self.model_config.max_cand_rules - len(rule_tars)
                        rule_tars.extend(num_pad * self.vocab_simple.encode(constant.SYMBOL_PAD))
                else:
                    rule_ids = rule_ids[:self.model_config.max_cand_rules]
                    rule_tars = rule_tars[:self.model_config.max_cand_rules]

            if 'rule' in self.model_config.memory:
                return np.array(rule_ids, dtype=np.int32), np.array(rule_tars, dtype=np.unicode)
            elif 'direct' in self.model_config.memory:
                return np.array(rule_ids, dtype=np.int32), np.array(rule_tars, dtype=np.int32)

        if self.model_config.subword_vocab_size and self.model_config.seg_mode:
            output_complex, output_simple, output_complex_seg, output_simple_seg = tf.py_func(
                process_line_pair,
                [features['line_comp'], features['line_simp']],
                [tf.int32, tf.int32, tf.int32, tf.int32],
                name='process_line_pair')
            output_complex.set_shape(
                [self.model_config.max_complex_sentence])
            output_simple.set_shape(
                [self.model_config.max_simple_sentence])
            output_complex_seg.set_shape(
                [self.model_config.max_complex_sentence])
            output_simple_seg.set_shape(
                [self.model_config.max_simple_sentence])
            output = {
                'line_comp_ids': output_complex,
                'line_simp_ids': output_simple,
                'line_comp_segids': output_complex_seg,
                'line_simp_segids': output_simple_seg,
            }
        else:
            output_complex, output_simple = tf.py_func(
                process_line_pair,
                [features['line_comp'], features['line_simp']],
                [tf.int32, tf.int32])
            output_complex.set_shape(
                [self.model_config.max_complex_sentence])
            output_simple.set_shape(
                [self.model_config.max_simple_sentence])
            output = {
                'line_comp_ids': output_complex,
                'line_simp_ids': output_simple,
            }

        if self.model_config.tune_style:
            if self.model_config.tune_style[0]:
                output['ppdb_score'] = features['ppdb_score']
            if self.model_config.tune_style[1]:
                output['dsim_score'] = features['dsim_score']
            if self.model_config.tune_style[2]:
                output['add_score'] = features['add_score']
            if self.model_config.tune_style[3]:
                output['len_score'] = features['len_score']

        if 'rule' in self.model_config.memory or 'direct' in self.model_config.memory:
            if 'rule' in self.model_config.memory:
                rule_id, rule_tars = tf.py_func(
                    process_rule, [features['ppdb_rule']], [tf.int32, tf.string],
                    name='process_rule')
            elif 'direct' in self.model_config.memory:
                rule_id, rule_tars = tf.py_func(
                    process_rule, [features['ppdb_rule']], [tf.int32, tf.int32],
                    name='process_rule')

            rule_id.set_shape([self.model_config.max_cand_rules])
            rule_tars.set_shape([self.model_config.max_cand_rules])
            output['rule_id'] = rule_id
            output['rule_target'] = rule_tars

        return output

    def _get_dataset(self, path):
        dataset = tf.data.TFRecordDataset([path]).repeat().shuffle(10000)
        dataset = dataset.prefetch(self.model_config.batch_size * 100)
        dataset = dataset.map(self._parse, num_parallel_calls=50)
        return dataset.batch(self.model_config.batch_size)
