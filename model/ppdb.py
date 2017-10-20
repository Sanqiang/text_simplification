from model.model_config import get_path, WikiDressLargeDefault
from data_generator.vocab import Vocab
from collections import defaultdict
from model.model_config import DefaultTrainConfig, WikiDressLargeTrainConfig
from util import constant

from util.arguments import get_args
import random as rd
from numpy.random import choice
from nltk.stem import WordNetLemmatizer
import re
import time


args = get_args()


class PPDB:
    def __init__(self, model_config):
        self.model_config = model_config
        self.wnl = WordNetLemmatizer()
        # rules is in structure
        # ori_words => tag => list of [tar words, its weight]
        self.rules = defaultdict(dict)
        self.populate_ppdb()

    def populate_ppdb(self):
        for line in open(self.model_config.path_ppdb_refine, encoding='utf-8'):
            items = line.split('\t')
            weight = float(items[1])
            tag = items[2]
            ori_words = items[3].strip()
            tar_words = items[4].strip()
            if re.match('[a-zA-Z]+', ori_words) or re.match('[a-zA-Z]+', tar_words):
                if ori_words not in self.rules:
                    self.rules[ori_words] = {}
                if tag not in self.rules[ori_words]:
                    if tag[0] == '[':
                        tag = tag[1:]
                    if tag[-1] == ']':
                        tag = tag[:-1]
                    self.rules[ori_words][tag] = []
                self.rules[ori_words][tag].append([tar_words, weight])
        # # Normalize the weight
        # for ori_words in self.rules:
        #     for tag in self.rules[ori_words]:
        #         norm = sum([p[1] for p in self.rules[ori_words][tag]]) + 1. * 1e7
        #         self.rules[ori_words][tag] = [[p[0], p[1]/norm] for p in self.rules[ori_words][tag]]


    """generate_synatx and recursive_gen are through NLTK called Stanford NLP tools
       for faster processing, we use java. script/SyntaxParser.java"""
    # def generate_synatx(self, syntax):
    #     outout_list = []
    #     for tree in syntax:
    #         if tree.label() == 'ROOT':
    #             tree = tree[0]
    #         try:
    #             self.recursive_gen(tree, outout_list)
    #         except Exception:
    #             print(syntax)
    #     outout_list = '\t'.join(outout_list)
    #     return outout_list
    #
    # def recursive_gen(self, tree, outout_list):
    #     if type(tree[0]) == str and len(tree) == 1:
    #         word = tree[0]
    #         if '@' not in word:
    #             label = tree.label()
    #             outout_list.append('%s=>%s' % (label, word))
    #         return word
    #
    #     output = []
    #     for node in tree:
    #         node_str = self.recursive_gen(node, outout_list)
    #         output.append(node_str)
    #     words = ' '.join(output)
    #     if '@' not in words:
    #         label = tree.label()
    #         outout_list.append('%s=>%s' % (label, words))
    #     return words

    def process_training(self, line_sep='\n'):
        output = ''
        line_idx = 0
        syntaxs = open(self.model_config.train_dataset_simple_syntax, encoding='utf-8').readlines()
        pre_time = time.time()
        for i, syntax in enumerate(syntaxs):
            syntax = syntax.strip()
            rules = ''
            if len(syntax) > 0:
                syntax_pairs = [p.split('=>') for p in syntax.split('\t')]
                rules = ppdb.process_sent(syntax_pairs)
            rules = '\t'.join(rules)
            output = line_sep.join([output, rules])
            line_idx += 1
            if line_idx % 1000 == 0:
                cur_time = time.time()
                print('processed %s. in %s' % (line_idx, cur_time - pre_time))
                pre_time = cur_time

        output = output[len(line_sep):]

        f = open(self.model_config.train_dataset_simple_ppdb, 'w', encoding='utf-8')
        f.write(output)
        f.close()

    def process_sent(self, syntax_pairs):
        # PPDB doesn't have am is are but only be
        filter_rules = []
        for tag, words in syntax_pairs:
            words = words.lower()
            words = words.replace(' am ', ' be ')
            words = words.replace(' is ', ' be ')
            words = words.replace(' are ', ' be ')

            if words in self.rules and (tag in self.rules[words] or 'X' in self.rules[words]):
                if tag in self.rules[words]:
                    filter_rules.append('%s=>%s' % (tag, words))

                if 'x' in self.rules[words]:
                    filter_rules.append('%s=>%s' % ('x', words))
        return filter_rules

    def simplify(self, sent, pairs, vocab):
        sent = ' '.join(sent)
        if not pairs:
            return None, None

        def isplural(word):
            lemma = self.wnl.lemmatize(word, 'n')
            plural = True if word is not lemma else False
            return plural, lemma

        def ori2be(sent, words):
            be = None
            rule_ori_list = words.split()
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
                raise Exception('be parsing error: %s\n%s' % (sent, words))
            return sent, be

        def be2ori(sent, target, be):
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

        num_trials = 5
        while True:
            num_trials -= 1
            if num_trials == 0:
                return None, None

            idx = int(rd.random() * len(pairs))
            pair = pairs[idx]
            if len(pair) != 2:
                print('pairs error! %s in %s.' % (pairs, sent))
                continue
            words = pair[1]
            tag = pair[0]
            # Use originl (includes be) to check target
            target_pairs = self.rules[words][tag]
            if 'X' in self.rules[words]:
                target_pairs += self.rules[words]['X']
            # Use replaced one (be will changed to am,is,are,be) to check replace
            sent, be = ori2be(sent, words)

            rule_tars = [p[0] for p in target_pairs]
            rule_tars_p_norm = sum([p[1] for p in target_pairs])
            rule_tars_p = [p[1] / rule_tars_p_norm for p in target_pairs]
            if not rule_tars:
                print('empty rule_tars')
                continue
            target = choice(rule_tars, p=rule_tars_p, size=1)[0]
            target_p = rule_tars_p[rule_tars.index(target)]

            target = be2ori(sent, target, be)
            nsent = sent.lower().replace(words, target)

            target_list = target.split()
            nsent_idx = []
            nsent_weight = []
            no_unk = True
            for word in nsent.split():
                wid = vocab.encode(word)
                if wid == vocab.encode(constant.SYMBOL_UNK):
                    no_unk = False
                    break
                nsent_idx.append(wid)
                if word in target_list:
                    nsent_weight.append(target_p)
                else:
                    nsent_weight.append(1.0)

            if no_unk:
                break
            else:
                continue

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
        tag = items[2]
        if tag == '[CD]':
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