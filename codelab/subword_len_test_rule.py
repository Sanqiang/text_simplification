from data_generator.vocab import Vocab
from util import constant
from types import SimpleNamespace
from collections import Counter
import os
import json

model_config = SimpleNamespace(min_count=0, subword_vocab_size=0, lower_case=True, bert_mode=['bert_token'], top_count=9999999999999)
vocab = Vocab(model_config, '/zfs1/hdaqing/saz31/dataset/vocab/bert/vocab_30k')
base = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/features/'
max_len = 0
for file in os.listdir(base):
    f = open(base + file)
    for line in f:
        obj = json.loads(line)
        rule = obj['ppdb_rule'].split('\t')

        for pair in rule:
            items = pair.split('=>')
            if len(items) != 3 or '-' in pair or '\'' in pair or '"' in pair or ',' in pair:
                continue
            words = items[1].lower().split()
            words = vocab.encode(' '.join(words))
            if len(words) >= max_len:
                print(pair)
                max_len = len(words)
            # max_len = max(max_len, len(words))

    print('cur max_len:%s with file %s' % (max_len, file))

print(max_len)