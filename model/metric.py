import numpy as np
from util import constant
from util.sari import SARIsent
from util.fkgl import get_fkgl
from util.decode import truncate_sent
from model.lm import GoogleLM
from model.ppdb import PPDB

from nltk.translate.bleu_score import sentence_bleu


class Metric:
    def __init__(self, model_config, data):
        self.model_config = model_config
        self.data = data
        # if model_config.use_quality_model:
        #     self.lm = GoogleLM()
        # if model_config.rl_simp:
        #     self.ppdb = PPDB(model_config)
        #     # Copied from data_generator/train_data.py
        #     self.syntaxs = []
        #     for line in open(self.model_config.train_dataset_complex_ppdb, encoding='utf-8'):
        #         syntax = [r.split('=>') for r in line.strip().split('\t') if len(r) > 0]
        #         self.syntaxs.append(syntax)

    """Used for training weight."""
    """RL version 1: automatically pick up words and give weight"""
    # todo(sanqiang): reword function need to improved
    def rl_process(self, sentence_complex_input, sentence_simple_input, sentence_generation):
        batch_size = np.shape(sentence_simple_input)[0]
        nsents = []
        nsent_weights = []
        for batch_i in range(batch_size):

            cur_sentence_simple_input_list = [self.data.vocab_simple.describe(wid)
                                              for wid in list(sentence_simple_input[batch_i])
                                              if wid >= constant.REVERED_VOCAB_SIZE]
            cur_sentence_simple_input_str = truncate_sent(' '.join(cur_sentence_simple_input_list))
            cur_sentence_generation_list = [self.data.vocab_simple.describe(wid)
                                            for wid in list(sentence_generation[batch_i])
                                            if wid >= constant.REVERED_VOCAB_SIZE]
            cur_sentence_generation_str = truncate_sent(' '.join(cur_sentence_generation_list))
            cur_sentence_complex_input_list = [self.data.vocab_complex.describe(wid)
                                               for wid in list(sentence_complex_input[batch_i])
                                               if wid >= constant.REVERED_VOCAB_SIZE]
            cur_sentence_complex_input_str = truncate_sent(' '.join(cur_sentence_complex_input_list))

            nsent = list(cur_sentence_simple_input_list[:(self.model_config.rl_prelenth)])
            nsent_weight = [1.0] * self.model_config.rl_prelenth
            cand_words = [w for w in cur_sentence_simple_input_list]
            for w in nsent:
                cand_words.remove(w)
            while True:
                cand_words_score = [0] * len(cand_words)
                for cand_idx in range(len(cand_words)):
                    cand_words_score[cand_idx] = self.caculate_score(
                        cur_sentence_complex_input_list, cur_sentence_simple_input_list, cur_sentence_generation_list,
                        cur_sentence_complex_input_str, cur_sentence_simple_input_str, cur_sentence_generation_str,
                        nsent, cand_words, cand_idx)
                cand_idx_max = int(np.argmax(cand_words_score))
                word_max = cand_words[cand_idx_max]
                word_score_max = cand_words_score[cand_idx_max]
                if word_score_max > 1.0:
                    nsent.append(word_max)
                    nsent_weight.append(word_score_max)
                    cand_words.remove(word_max)
                elif len(nsent) < len(cur_sentence_simple_input_list):
                    word_max = cur_sentence_simple_input_list[len(nsent)]
                    nsent.append(word_max)
                    nsent_weight.append(1.0)
                    if word_max in cand_words:
                        cand_words.remove(word_max)
                else:
                    break

                if len(nsent) + 1 > self.model_config.max_simple_sentence or not cand_words:
                    break
            nsent.insert(0, constant.SYMBOL_START)
            nsent_weight.insert(0, 1.0)
            if len(nsent) + 1 < self.model_config.max_simple_sentence:
                nsent.append(constant.SYMBOL_END)
                nsent_weight.append(1.0)
            while len(nsent) < self.model_config.max_simple_sentence:
                nsent.append(constant.SYMBOL_PAD)
                nsent_weight.append(0.0)
            # print(cur_sentence_simple_input_list)
            # print(nsent)
            # print([p for p in nsent_weight])
            # print('==============================================')
            nsent = [self.data.vocab_simple.encode(w) for w in nsent]
            nsents.append(nsent)
            nsent_weights.append(nsent_weight)
        return np.array(nsents, dtype=np.int32), np.array(nsent_weights, dtype=np.float32)

    def caculate_score(self,
                       sentence_complex_input_list, sentence_simple_input_list, sentence_generation_list,
                       sentence_complex_input_str, sentence_simple_input_str, sentence_generation_str,
                       nsent, cand_words, cand_idx):
        weights = []
        sentence_cand_list = nsent + [cand_words[cand_idx]]
        sentence_cand_str = ' '.join(sentence_cand_list)

        if self.model_config.rl_sari:
            try:
                weight = self.model_config.rl_sari * SARIsent(
                    sentence_complex_input_str,
                    sentence_cand_str,
                    [sentence_simple_input_str])
                weights.append(weight)
            except ZeroDivisionError:
                print('ignore sari weight: %s' % sentence_cand_str)

        if self.model_config.rl_fkgl:
            try:
                weight = self.model_config.rl_fkgl * get_fkgl(sentence_cand_str)
                weights.append(weight)
            except ZeroDivisionError:
                print('ignore fkgl weight: %s' % sentence_cand_str)

        if self.model_config.rl_bleu:
            try:
                weight = sentence_bleu([sentence_simple_input_list], sentence_cand_list, [1, 1])
                weights.append(weight)
                # print('%s\n%s\n%s\n======\n' % (sentence_cand_list, sentence_simple_input_list, weight))
            except ZeroDivisionError:
                print('ignore bleu weight: %s' % sentence_cand_str)
        return 1.0 + np.mean(weights)

    """RL version 2: only weight words"""
    def rl_process2(self, sentence_complex_input, sentence_simple_input, sentence_generation, idxs):
        batch_size = np.shape(sentence_simple_input)[0]
        nsent_weights = []
        for batch_i in range(batch_size):
            idx = idxs[batch_i]
            syntax = self.syntaxs[idx]
            cur_sentence_generation_list = [self.data.vocab_simple.describe(wid)
                                            for wid in list(sentence_generation[batch_i])]
            prob_mapper = {}
            nsent_weight = []
            if self.model_config.rl_simp:
                for quad in syntax:
                    target_words = quad[2]
                    for target_word in target_words.split():
                        prob_mapper[target_word] = self.model_config.rl_simp
                    # print('==========Debug Info: Simp==========')
                    # print('target_word')
                    # print(target_word)

            if self.model_config.rl_keep:
                cur_sent_simp_set = set(cur_sentence_generation_list)
                cur_sent_somp_set = set([self.data.vocab_complex.describe(wid)
                                               for wid in list(sentence_complex_input[batch_i])
                                               if wid >= constant.REVERED_VOCAB_SIZE])
                # print('==========Debug Info: Keep==========')
                # print('cur_sent_simp_set & cur_sent_somp_set')
                # print(cur_sent_simp_set & cur_sent_somp_set)
                for word in (cur_sent_simp_set & cur_sent_somp_set):
                    prob_mapper[word] = self.model_config.rl_keep

            for word in cur_sentence_generation_list:
                if word in prob_mapper:
                    nsent_weight.append(prob_mapper[word])
                else:
                    nsent_weight.append(1.0)

            nsent_weights.append(nsent_weight)
            # print('==========Debug Info==========')
            # print('idx')
            # print(idx)
            # print('prob_mapper')
            # print(prob_mapper)
            # print('sentence_complex_input')
            # print(sentence_complex_input)
            # print('sentence_simple_input')
            # print(sentence_simple_input)
            # print('sentence_generation')
            # print(sentence_generation)
            # print('syntax')
            # print(syntax)

        return np.array(sentence_simple_input, dtype=np.int32), np.array(nsent_weights, dtype=np.float32)

    """Used for Data quality."""
    def bleu_quality(self, sentence_simple_input, sentence_complex_input):
        batch_size = np.shape(sentence_simple_input)[0]
        bleus = []

        for batch_i in range(batch_size):
            sent_simple = [self.data.vocab_simple.describe(wid)
                           for wid in sentence_simple_input[batch_i, :]
                           if wid != self.data.vocab_simple.encode(constant.SYMBOL_PAD)]
            sent_complex = [self.data.vocab_complex.describe(wid)
                           for wid in sentence_complex_input[batch_i, :]
                            if wid != self.data.vocab_complex.encode(constant.SYMBOL_PAD)]
            try:
                bleu = sentence_bleu(sent_simple, sent_complex)
            except:
                bleu = 0
            bleus.append(bleu)

        return np.array(bleus, dtype=np.float32)

    def length_ratio(self, sentence_simple_input, sentence_complex_input):
        batch_size = np.shape(sentence_simple_input)[0]
        len_ratios = []

        for batch_i in range(batch_size):
            sent_simple = [wid for wid in sentence_simple_input[batch_i, :]
                           if wid != self.data.vocab_simple.encode(constant.SYMBOL_PAD)
                           and wid != self.data.vocab_simple.encode(constant.SYMBOL_UNK)]
            sent_complex = [wid for wid in sentence_complex_input[batch_i, :]
                            if wid != self.data.vocab_complex.encode(constant.SYMBOL_PAD)
                            and wid != self.data.vocab_complex.encode(constant.SYMBOL_UNK)]

            ratio = (len(sent_simple) - 1) / (len(sent_complex) - 1)
            len_ratios.append(ratio)

        return np.array(len_ratios, dtype=np.float32)

    def lm_quality(self, sentence_simple_input):
        batch_size = np.shape(sentence_simple_input)[0]
        lm_weights = []

        for batch_i in range(batch_size):
            sent_simple = [self.data.vocab_simple.describe(wid)
                           for wid in sentence_simple_input[batch_i, :]]
            sent_simple = truncate_sent(sent_simple)
            sent_simple = ' '.join(sent_simple)
            lm_weight = self.lm.get_weight(sent_simple)
            lm_weights.append(lm_weight)

        return np.array(lm_weights, dtype=np.float32)


if __name__ == '__main__':
    from model.model_config import DefaultConfig
    from data_generator.train_data import TrainData
    sentence_complex_input = [[3, 16, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 16, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    sentence_simple_input = [[3, 11, 12 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 11, 12 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    sentence_generation = [[3, 11, 12 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 11, 12 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    config = DefaultConfig()
    config.rl_bleu = 1.0
    config.rl_fkgl = 1.5
    config.rl_sari = 3.0
    metric = Metric(config, TrainData(config))
    metric.rl_process(sentence_complex_input, sentence_simple_input, sentence_generation)
