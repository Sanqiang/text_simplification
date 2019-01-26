"""
Generator PPDB rule vocab
"""
import json
from collections import defaultdict
import os
import operator

c = defaultdict(float)
PATHS = [
    '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/features/',
    '/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/ner/features/']

for path in PATHS:
    for file in os.listdir(path):
        cpath = os.path.join(path, file)
        for line in open(cpath):
            obj = json.loads(line)
            ppdb_rules = obj['ppdb_rule'].split('\t')
            for ppdb_rule in ppdb_rules:
                rule_str = '=>'.join(ppdb_rule.split('=>')[:2])
                if len(rule_str) > 0:
                    weight = float(ppdb_rule.split('=>')[2])
                    if rule_str in c:
                        c[rule_str] = max(weight, c[rule_str])
                    else:
                        c[rule_str] = weight

c_sorted = sorted(c.items(), key=operator.itemgetter(1), reverse=True)
vocab_str = []
for wd, cnt in c_sorted:
    vocab_str.append('%s\t%s' % (wd , str(cnt)))
open('/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/rule_vocab', 'w').write('\n'.join(vocab_str))




