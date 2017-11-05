"""

"""
from model.lm import GoogleLM
from collections import defaultdict
import re


class DataPrepareBase:
    def __init__(self, model_config):
        self.model_config = model_config
        self.language_model = GoogleLM()

    def init_ppdb(self):
        return None

    def init_ppdb_simple(self):
        self.ppdb_simple_rules = defaultdict(dict)
        for line in open(self.model_config.path_ppdb_refine, encoding='utf-8'):
            items = line.split('\t')
            weight = float(items[1])
            ori_words = items[3].strip()
            tar_words = items[4].strip()
            tags = items[2]
            for tag in tags.split('\\'):
                # Tag in www.cs.cornell.edu/courses/cs474/2004fa/lec1.pdf
                if tag[0] == '[':
                    tag = tag[1:]
                if tag[-1] == ']':
                    tag = tag[:-1]

                if re.match('[a-zA-Z]+', ori_words) or re.match('[a-zA-Z]+', tar_words):
                    if ori_words not in self.ppdb_simple_rules:
                        self.ppdb_simple_rules[ori_words] = {}
                    if tag not in self.ppdb_simple_rules[ori_words]:
                        self.ppdb_simple_rules[ori_words][tag] = {}
                    if tar_words in self.ppdb_simple_rules[ori_words][tag]:
                        self.ppdb_simple_rules[ori_words][tag][tar_words] = max(weight, self.ppdb_simple_rules[ori_words][tag][tar_words])
                    else:
                        self.ppdb_simple_rules[ori_words][tag][tar_words] = weight

    def get_candidate_sent(self):
        self.init_ppdb_simple()
        syntaxs = open(self.model_config.train_dataset_complex_syntax, encoding='utf-8').readlines()
        sents_simp = open(self.model_config.train_dataset_simple, encoding='utf-8').readlines()
        sents_comp = open(self.model_config.train_dataset_complex, encoding='utf-8').readlines()
        for i, syntax in enumerate(syntaxs):
            sent_comp = sents_comp[i]
            rules = [r.split('=>') for r in syntax.strip().split('\t') if len(r) > 0]
            for rule in rules:
                origin_words = rule[1].lower()
                if origin_words in self.ppdb_simple_rules:
                    best_perplexity = self.language_model.get_weight(sent_comp)
                    best_target_words = None
                    for tag, target_words_list in self.ppdb_simple_rules[origin_words].items():
                        for target_words in target_words_list:
                            perplexity = self.language_model.get_weight(sent_comp.replace(origin_words, target_words))
                            if perplexity < best_perplexity:
                                best_perplexity = perplexity
                                best_target_words = target_words
                    if best_target_words is not None:
                        sent_comp = sent_comp.replace(origin_words, best_target_words)
            print('Process %s' % i)



if __name__ == '__main__':
    from model.model_config import WikiDressLargeDefault
    DataPrepareBase(WikiDressLargeDefault()).get_candidate_sent()




