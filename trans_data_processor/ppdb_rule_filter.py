"""
Generator PPDB rule vocab
"""
from nltk.corpus import stopwords
from collections import defaultdict
import pygtrie as trie
import operator

stopwords_set = set(stopwords.words('english'))
for line in open('/Users/sanqiang/git/ts/text_simplification/trans_data_processor/stopwords.list'):
    stopwords_set.add(line.strip())

RULE_VOCAB = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/rule_vocab'
RULE_VOCAB2 = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/rule_vocab2'
THRESHOLD = 0.0

# Populate rule vocab
dup_checker = defaultdict(set)
def _wd_valid(ori_wd):
    for wd in ori_wd.split():
        if wd not in stopwords_set:
            return True
    return False

def _populate_dup_checker(ori_wd, wds):
    for wd in wds.split():
        if wd not in stopwords_set:
            dup_checker[ori_wd].add(wd)

def _val_dup_checker(ori_wd, wds):
    for wd in wds.split():
        if wd not in stopwords_set and wd in dup_checker[ori_wd]:
            return False
    return True

nlines = []
rule_vocab = trie.StringTrie(separator=' ')
for line in open(RULE_VOCAB):
    items = line.split('\t')
    weight = float(items[1])
    if weight > THRESHOLD:
        pair = items[0].split('=>')
        ori_wd = pair[0]
        tar_wd = pair[1]
        if (tar_wd in stopwords_set or ori_wd in stopwords_set
                or ',' in ori_wd or '.' in ori_wd
                or '\'' in ori_wd or '"' in ori_wd
                or ',' in tar_wd or '.' in tar_wd
                or '\'' in tar_wd or '"' in tar_wd or
                not _wd_valid(ori_wd) or not _wd_valid(tar_wd)):
            continue
        if ori_wd not in rule_vocab:
            rule_vocab[ori_wd] = []
        if len(rule_vocab[ori_wd]) <= 5 and _val_dup_checker(ori_wd, tar_wd):
            rule_vocab[ori_wd].append(pair)
            _populate_dup_checker(ori_wd, tar_wd)
            nlines.append((line.strip(), weight))

nlines.sort(key=operator.itemgetter(1), reverse=True)
print('get filtered vocab with size %s' % len(nlines))
open(RULE_VOCAB2, 'w').write('\n'.join([it[0] for it in nlines]))