from util.data.text_encoder import SubwordTextEncoder
from util.data import text_encoder
from data_generator.vocab import Vocab
from model.model_config import WikiDressLargeDefault

from os import listdir
from collections import Counter

dict = {}
for line in open('/Users/zhaosanqiang916/git/text_simplification/data/dummy_vocab'):
    items = line.split('\t')
    word = items[0]
    cnt = int(items[1])
    dict[word] = cnt

c = Counter(dict)
output_path = "/Users/zhaosanqiang916/git/text_simplification/data/dummy_subvocab"
sub_word = SubwordTextEncoder.build_to_target_size(5, c, 5, 1e3,
                                                               num_iterations=10)
for i, subtoken_string in enumerate(sub_word._all_subtoken_strings):
    if subtoken_string in text_encoder.RESERVED_TOKENS_DICT:
        sub_word._all_subtoken_strings[i] = subtoken_string + "_"
sub_word.store_to_file(output_path)