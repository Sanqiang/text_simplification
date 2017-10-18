from model.model_config import get_path, WikiDressLargeDefault
from data_generator.vocab import Vocab
from collections import defaultdict
from model.model_config import DefaultTrainConfig, WikiDressLargeTrainConfig

import numpy as np
from queue import Queue
from nltk.parse.stanford import StanfordParser
from nltk.tree import Tree
from util.arguments import get_args
import random as rd
from numpy.random import choice
from nltk.stem import WordNetLemmatizer


args = get_args()


class PPDB:
    def __init__(self, model_config):
        self.model_config = model_config
        self.wnl = WordNetLemmatizer()
        # rules is in structure
        # ori_words => list of tar words along with its weight
        self.rules = defaultdict(dict)
        # rules_index is for indexing rules
        self.rules_index = defaultdict(set)
        self.populate_ppdb()

    def populate_ppdb(self):
        for line in open(self.model_config.path_ppdb_refine, encoding='utf-8'):
            items = line.split('\t')
            weight = float(items[1])
            ori_words = items[3].strip()
            tar_words = items[4].strip()
            if ori_words not in self.rules:
                self.rules[ori_words] = []
            self.rules[ori_words].append([tar_words, weight])
        # Normalize
        for ori_words in self.rules:
            tar_words_pairs = [[p[0], 0.0001 + p[1]]
                               for p in self.rules[ori_words]]
            tar_words_norm = sum([p[1] for p in tar_words_pairs])
            tar_words_pairs = [[p[0], p[1] / tar_words_norm]
                               for p in tar_words_pairs]
            tar_words_pairs = sorted(tar_words_pairs, key=lambda x:-x[1])
            self.rules[ori_words] = tar_words_pairs
        # Populate rules_index
        for ori_words in self.rules:
            ori_words_list = ori_words.split()
            if len(ori_words_list) > 0:
                word_index = ori_words_list[0]
                self.rules_index[word_index].add(ori_words)

    def process_training(self, line_sep='\n'):
        output = ''
        line_idx = 0
        for line in open(self.model_config.train_dataset_simple, encoding='utf-8'):
            line = line.lower()
            rules = ppdb.process_sent(line)
            rules = '\t'.join(rules)
            output = line_sep.join([output, rules])
            line_idx += 1
            if line_idx % 1000 == 0:
                print('processed %s.' % line_idx)
        output = output[len(line_sep):]

        f = open(self.model_config.train_dataset_simple_ppdb, 'w', encoding='utf-8')
        f.write(output)
        f.close()

    def process_sent(self, sent):
        # PPDB doesn't have am is are but only be
        sent = sent.replace(' am ', ' be ')
        sent = sent.replace(' is ', ' be ')
        sent = sent.replace(' are ', ' be ')

        rules = []
        words = sent.split()
        for word in words:
            postings = self.rules_index[word]
            for posting in postings:
                # Verified each posting
                # avoid sub word by adding space
                add_rule = ' ' + posting + ' ' in ' ' + sent + ' '
                if add_rule:
                    rules.append(posting)
        filter_rules = []
        for i in range(len(rules)):
            add_rule = True
            for j in range(len(rules)):
                if i != j and rules[i] in rules[j]:
                    add_rule = False
                    break
            if add_rule:
                filter_rules.append(rules[i])
        return filter_rules

    def simplify(self, sent, rule_oris, vocab):
        sent = ' '.join(sent)
        if not rule_oris:
            return None, None

        def isplural(word):
            lemma = self.wnl.lemmatize(word, 'n')
            plural = True if word is not lemma else False
            return plural, lemma

        def getbe_rule_ori(sent, rule_ori):
            be = None
            rule_ori_list = rule_ori.split()
            if 'be' in rule_ori_list:
                be_idx = rule_ori_list.index('be')
                rule_ori_list[be_idx] = 'is'
                be = 'is'
                if ' '.join(rule_ori_list) in sent:
                    return ' '.join(rule_ori_list), be
                rule_ori_list[be_idx] = 'are'
                be = 'are'
                if ' '.join(rule_ori_list) in sent:
                    return ' '.join(rule_ori_list), be
                rule_ori_list[be_idx] = 'am'
                be = 'am'
                if ' '.join(rule_ori_list) in sent:
                    return ' '.join(rule_ori_list), be
                rule_ori_list[be_idx] = 'be'
                be = 'be'
                if ' '.join(rule_ori_list) in sent:
                    return ' '.join(rule_ori_list), be
                raise Exception('be parsing error: %s\n%s' % (sent, rule_ori))
            return sent, be

        def be2actual(sent, target, be):
            target_list = target.split()
            if 'be' in target_list:
                be_idx = target_list.index('be')
                if be is None:
                    sent_list = sent.split()
                    if 'are' in sent_list:
                        be = 'are'
                    elif 'am' in sent_list:
                        be = 'am'
                    elif 'is' in sent_list:
                        be = 'is'
                    else:
                        need_replaced = True
                        for word in sent.split():
                            if word == 'and' or isplural(word):
                                be = 'are'
                                need_replaced = False
                                break
                        if need_replaced:
                            be = 'is'
                target_list[be_idx] = be
                return ' '.join(target_list)
            return target

        idx = int(rd.random() * len(rule_oris))
        rule_ori = rule_oris[idx]
        # Use originl (includes be) to check target
        rule_tars_pairs = self.rules[rule_ori]
        # Use replaced one (be will changed to am,is,are,be) to check replace
        sent, be = getbe_rule_ori(sent, rule_ori)

        rule_tars = [p[0] for p in rule_tars_pairs]
        rule_tars_p = [p[1] for p in rule_tars_pairs]
        if not rule_tars:
            print('emoty rule_tars')
        target = choice(rule_tars, p=rule_tars_p, size=1)[0]
        target_p = rule_tars_p[rule_tars.index(target)]

        target = be2actual(sent, target, be)
        nsent = sent.lower().replace(rule_ori, target)

        target_list = target.split()
        nsent_idx = []
        nsent_weight = []
        for word in nsent.split():
            wid = vocab.encode(word)
            nsent_idx.append(wid)
            if word in target_list:
                nsent_weight.append(1.0 + target_p)
            else:
                nsent_weight.append(1.0)

        return nsent_idx, nsent_weight


def get_refine_data():
    """Keep the rules that only relevant to our vocab so that speedup the processing."""
    model_config = WikiDressLargeDefault()
    vocab = Vocab(model_config,
                  get_path(
                      '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.vocab.lower'))

    path_ppdb = get_path(
        '../text_simplification_data/ppdb/SimplePPDB')
    path_ppdb_refine = get_path(
        '../text_simplification_data/ppdb/ppdb_rules2.simp.rules')

    noutputs = []
    line_idx = 0
    f_simp = open(path_ppdb_refine, 'w', encoding='utf-8')
    for line in open(path_ppdb, encoding='utf-8'):
        line_idx += 1
        if line_idx % 1000 == 0:
            print(line_idx)
        items = line.split('\t')
        pos = items[2]
        if pos == '[CD]':
            continue
        ori_words = items[3]
        tar_words = items[4]
        ori_words_list = ori_words.split()
        tar_words_list = tar_words.split()
        if ori_words != tar_words and (
                    any([word for word in ori_words_list if vocab.contain(word)]) or
                    any([word for word in tar_words_list if vocab.contain(word)])):
            noutputs.append(line.strip())

    for line in noutputs:
        f_simp.write(line)
        f_simp.write('\n')
    f_simp.flush()
    f_simp.close()


if __name__ == '__main__':
    config = None
    if args.mode == 'dummy':
        config = DefaultTrainConfig()
    elif args.mode == 'dress':
        config = WikiDressLargeTrainConfig()
    # get_refine_data()
    ppdb = PPDB(config)
    ppdb.process_training()
    # ppdb.simplify(
    #     'There is manuscript evidence that PERSON@1 continued to work on these pieces as late as the period NUMBER@1 â '' NUMBER@2 , and that her niece and nephew , PERSON@2 and PERSON@3 , made further additions as late as NUMBER@3 .',
    #     'manuscript	evidence that	continued	to work on	the period	niece	further')