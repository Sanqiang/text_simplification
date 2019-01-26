"""Provide vocabulary"""
from collections import Counter
import json
from os.path import exists
from trans_data_processor.vocab_util import SubwordTextEncoder
from trans_data_processor import vocab_util

PATH_PREFIX_FEATURES = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/features/shard'
PATH_PREFIX_FEATURES2 = '/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/ner/features/shard'
PATH_VOCAB_COMP = '/zfs1/hdaqing/saz31/dataset/vocab/comp.vocab'
PATH_VOCAB_SIMP = '/zfs1/hdaqing/saz31/dataset/vocab/simp.vocab'
PATH_SUBVOCAB_COMP = '/zfs1/hdaqing/saz31/dataset/vocab/comp30k.subvocab'
PATH_SUBVOCAB_SIMP = '/zfs1/hdaqing/saz31/dataset/vocab/simp30k.subvocab'

c_comp, c_simp = Counter(), Counter()

for shard_id in range(1280):
    if not exists(PATH_PREFIX_FEATURES + str(shard_id)):
        continue
    lines_features = open(PATH_PREFIX_FEATURES + str(shard_id)).readlines()
    for line_features in lines_features:
        obj = json.loads(line_features)
        line_comp = obj['line_comp']
        line_simp = obj['line_simp']
        c_comp.update(line_comp.split())
        c_simp.update(line_simp.split())

for shard_id in range(15):
    if not exists(PATH_PREFIX_FEATURES + str(shard_id)):
        continue
    lines_features = open(PATH_PREFIX_FEATURES + str(shard_id)).readlines()
    for line_features in lines_features:
        obj = json.loads(line_features)
        line_comp = obj['line_comp']
        line_simp = obj['line_simp']
        c_comp.update(line_comp.split())
        c_simp.update(line_simp.split())

vocab_comps = []
for w, c in c_comp.most_common():
    vocab_comps.append('%s\t%s' % (w, c))
open(PATH_VOCAB_COMP, 'w').write('\n'.join(vocab_comps))

vocab_simps = []
for w, c in c_simp.most_common():
    vocab_simps.append('%s\t%s' % (w, c))
open(PATH_VOCAB_SIMP, 'w').write('\n'.join(vocab_simps))

print('Created Vocab.')

sub_word_comp_feeder = {}
for line in open(PATH_VOCAB_COMP):
    items = line.split('\t')
    word = items[0]
    cnt = int(items[1])
    sub_word_comp_feeder[word] = cnt

c_comp = Counter(sub_word_comp_feeder)
sub_word_comp = SubwordTextEncoder.build_to_target_size(
    30000, c_comp, 1, 1e5, num_iterations=10)
for i, subtoken_string in enumerate(sub_word_comp._all_subtoken_strings):
    if subtoken_string in vocab_util.RESERVED_TOKENS_DICT:
        sub_word_comp._all_subtoken_strings[i] = subtoken_string + "_"
sub_word_comp.store_to_file(PATH_SUBVOCAB_COMP)

sub_word_simp_feeder = {}
for line in open(PATH_VOCAB_SIMP):
    items = line.split('\t')
    word = items[0]
    cnt = int(items[1])
    sub_word_simp_feeder[word] = cnt

c_simp = Counter(sub_word_simp_feeder)
sub_word_simp = SubwordTextEncoder.build_to_target_size(
    30000, c_simp, 1, 1e5, num_iterations=10)
for i, subtoken_string in enumerate(sub_word_simp._all_subtoken_strings):
    if subtoken_string in vocab_util.RESERVED_TOKENS_DICT:
        sub_word_simp._all_subtoken_strings[i] = subtoken_string + "_"
sub_word_simp.store_to_file(PATH_SUBVOCAB_SIMP)

print('Created Subvocab.')


