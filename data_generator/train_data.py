import copy as cp
import random as rd

import numpy as np
from nltk import word_tokenize

from data_generator.vocab import Vocab
from model.ppdb import PPDB
from util import constant


class TrainData:
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
            self.data_simple, _ = self.populate_data(
                data_simple_path, self.vocab_simple, need_raw=False)
            self.data_complex, _ = self.populate_data(
                data_complex_path, self.vocab_complex, need_raw=False)
            assert len(self.data_complex) == self.size
            assert len(self.data_simple) == self.size
        else:
            self.data_it = self.get_data_sample_it(data_simple_path, data_complex_path)

        print('Use Train Dataset: \n Simple\t %s. \n Complex\t %s. \n Size\t %d.'
              % (data_simple_path, data_complex_path, self.size))

        if self.model_config.ppdb_mode != 'none':
            self.ppdb = PPDB(model_config)
            self.ppdb_simp_rules = self.populate_ppdb(self.model_config.train_dataset_simple_ppdb)
            self.ppdb_comp_rules = self.populate_ppdb(self.model_config.train_dataset_complex_ppdb)
            assert len(self.ppdb_simp_rules) == self.size
            assert len(self.ppdb_comp_rules) == self.size

        self.init_pretrained_embedding()

    def populate_ppdb(self, data_path):
        rules = []
        for line in open(data_path, encoding='utf-8'):
            rule = [r.split('=>') for r in line.strip().split('\t') if len(r) > 0]
            rules.append(rule)
        return rules

    def process_line(self, line, vocab, need_raw=False):
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

        return words, words_raw

    def get_size(self, data_complex_path):
        return len(open(data_complex_path, encoding='utf-8').readlines())

    def get_data_sample_it(self, data_simple_path, data_complex_path):
        f_simple = open(data_simple_path, encoding='utf-8')
        f_complex = open(data_complex_path, encoding='utf-8')
        i = 0
        while True:
            if i == self.size:
                f_simple = open(data_simple_path, encoding='utf-8')
                f_complex = open(data_complex_path, encoding='utf-8')
                i = 0
            line_complex = f_complex.readline()
            line_simple = f_simple.readline()
            words_complex, _ = self.process_line(line_complex, self.vocab_complex)
            words_simple, words_raw_simple = self.process_line(line_simple, self.vocab_simple, need_raw=True)

            yield i, words_simple, words_complex, cp.deepcopy([1.0] * len(words_simple)), cp.deepcopy([1.0] * len(words_complex))

            i += 1


    def populate_data(self, data_path, vocab, need_raw=False):
        # Populate data into memory
        data = []
        data_raw = []
        # max_len = -1
        # from collections import Counter
        # len_report = Counter()
        for line in open(data_path, encoding='utf-8'):
            words, words_raw = self.process_line(line, vocab, need_raw)
            if need_raw:
                data_raw.append(words_raw)
            data.append(words)
            # len_report.update([len(words)])
            # if len(words) > max_len:
            #     max_len = len(words)
        # print('Max length for data %s is %s.' % (data_path, max_len))
        # print('counter:%s' % len_report)
        return data, data_raw

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
            assert self.vocab_complex.vocab_size() ==  random_cnt + pretrained_cnt
            print(
                'For Vocab Complex, %s words initialized with pretrained vector, '
                'other %s words initialized randomly. Save to %s.' %
                (pretrained_cnt, random_cnt, self.model_config.pretrained_embedding_complex))
            # np.save(self.model_config.pretrained_embedding_complex, self.pretrained_emb_complex)

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
                'other %s words initialized randomly. Save to %s.' %
                (pretrained_cnt, random_cnt, self.model_config.pretrained_embedding_simple))
            # np.save(self.model_config.pretrained_embedding_simple, self.pretrained_emb_simple)

            del self.glove

    def get_data_sample(self):
        i = rd.sample(range(self.size), 1)[0]
        if self.model_config.ppdb_mode != 'none':
            data_simple, data_weight, attn_weight = self.ppdb.simplify(
                self.data_simple[i], self.data_complex[i],
                self.ppdb_simp_rules[i], self.ppdb_comp_rules[i],
                self.vocab_simple, self.vocab_complex)
            if data_simple:
                return i, data_simple, cp.deepcopy(self.data_complex[i]), data_weight, attn_weight

        return i, cp.deepcopy(self.data_simple[i]), cp.deepcopy(self.data_complex[i]), cp.deepcopy([1.0] * len(self.data_simple[i])), cp.deepcopy([1.0] * len(self.data_complex[i]))