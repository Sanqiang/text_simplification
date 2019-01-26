"""
Generate rules for eval/test
"""
from nltk.corpus import stopwords
from collections import defaultdict
import pygtrie as trie

stopwords_set = set(stopwords.words('english'))
for line in open('/Users/sanqiang/git/ts/text_simplification/trans_data_processor/stopwords.list'):
    stopwords_set.add(line.strip())

THRESHOLD = -999999999.0
RULE_VOCAB = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/rule_vocab'
COMP_PATHS = [
    '/Users/sanqiang/git/ts/text_simplification_data/test2/ncomp/test.8turkers.tok.norm',
    '/Users/sanqiang/git/ts/text_simplification_data/val2/ncomp/tune.8turkers.tok.norm']

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

for path in COMP_PATHS:
    outputs = []
    for line in open(path):
        output = []
        wds = line.split()
        for wid, wd in enumerate(wds):
            try:
                tar_unigrams = ['=>'.join(p) for p in rule_vocab.items(prefix=wd, shallow=True)[0][1]]
                output.extend(tar_unigrams)

                if wid+1 < len(wds):
                    bigram = wd + ' ' + wds[wid+1]
                    try:
                        tar_bigrams = ['=>'.join(p) for p in rule_vocab.items(prefix=bigram, shallow=True)[0][1]]
                        output.extend(tar_bigrams)

                        if wid+2 < len(wds):
                            trigram = bigram + ' ' + wds[wid+2]
                            try:
                                tar_trigram = ['=>'.join(p) for p in rule_vocab.items(prefix=trigram, shallow=True)[0][1]]
                                output.extend(tar_trigram)
                            except KeyError:
                                pass
                    except KeyError:
                        pass
            except KeyError:
                pass
        outputs.append('\t'.join(output))
    npath = path + '.ppdb'
    open(npath, 'w').write('\n'.join(outputs))





