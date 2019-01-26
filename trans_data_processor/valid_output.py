"""Valid before tensroflow.Example. i.e. 2nd tokenizer using spacy"""
import os
from os.path import exists
from datetime import datetime
from multiprocessing import Pool
import math
import json
import sys
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
import traceback


sys.path.insert(0,'/zfs1/hdaqing/saz31/dataset/tmp_trans/code')
# nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger', 'ner'])

is_trans = False
global_vocab = None

PATH_VOCAB_FREQ = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/freq'

if is_trans:
    # for trans
    PATH_FEATURES = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/features/shard'
    PATH_PREFIX_COMP = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/ncomp/shard'
    PATH_PREFIX_SIMP = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/nsimp/shard'
    PATH_PREFIX_PPDB = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/ppdb/shard'
    PATH_PREFIX_PPDBRULE = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/ppdb_rule/shard'
    PATH_PREFIX_META = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/meta/shard'
    PATH_PREFIX_STRUC = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/struc/shard'
else:
    # for wikilarge
    PATH_FEATURES = '/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/raw/features/shard'
    PATH_PREFIX_COMP = '/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/raw/ncomp/shard'
    PATH_PREFIX_SIMP = '/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/raw/nsimp/shard'
    PATH_PREFIX_PPDB = '/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/raw/ppdb/shard'
    PATH_PREFIX_PPDBRULE = '/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/raw/ppdb_rule/shard'
    PATH_PREFIX_META = '/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/raw/meta/shard'
    PATH_PREFIX_STRUC = '/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/raw/struc/shard'


# Validate Section method
from nltk.corpus import stopwords

stop_words_set = set(stopwords.words('english'))

ner_set = set()
for label in ['num', 'spe', 'location', 'organization', 'person', 'money', 'percent', 'date', 'time']:
    for i in range(0, 10):
        ner_set.add(label + str(i))


def _validate(line_ncomp, line_nsimp, ppdb_score, line_ppdb_rule, line_meta, dsim):
    set_comp = set([w for w in line_ncomp.split() if w in ner_set])
    set_simp = set([w for w in line_nsimp.split() if w in ner_set])
    # # Ignore same sentence
    if line_ncomp.strip() == line_nsimp.strip():
        return False

    if len(line_ncomp.split()) <= 5 or len(line_nsimp.split()) <= 5:
        return False

    if not line_ppdb_rule or ppdb_score <= 0:
        return False

    if dsim == 0.0:
        return False

    if is_trans:
        set_comp, set_simp = set(line_ncomp.split()), set(line_nsimp.split())
        extra_words_comp = set_comp - set_simp
        extra_words_simp = set_simp - set_comp
        score = 0.0
        for wd in extra_words_comp:
            score += global_vocab[wd]
        for wd in extra_words_simp:
            score -= global_vocab[wd]
        return score > 0
    else:
        return True


def generate_example(shard_id):
    cnt = 0
    if not exists(PATH_PREFIX_COMP + str(shard_id)) or not exists(PATH_PREFIX_SIMP + str(shard_id)):
        return
    if exists(PATH_FEATURES + str(shard_id)):
        return
    writer = open(PATH_FEATURES + str(shard_id), 'w')
    print('Start shard id %s' % shard_id)
    s_time = datetime.now()
    lines_comp = open(PATH_PREFIX_COMP + str(shard_id)).readlines()
    lines_simp = open(PATH_PREFIX_SIMP + str(shard_id)).readlines()
    lines_ppdb = open(PATH_PREFIX_PPDB + str(shard_id)).readlines()
    lines_ppdb_rule = open(PATH_PREFIX_PPDBRULE + str(shard_id)).readlines()
    lines_meta = open(PATH_PREFIX_META + str(shard_id)).readlines()
    lines_struc = open(PATH_PREFIX_STRUC + str(shard_id)).readlines()

    if not len(lines_comp) == len(lines_simp) == len(lines_ppdb) == len(lines_meta) == len(lines_struc):
        print('Error idx:%s with lens:%s, %s, %s, %s, %s' %
              (shard_id, len(lines_comp), len(lines_simp), len(lines_ppdb), len(lines_meta), len(lines_struc)))
        os.remove(PATH_FEATURES + str(shard_id))
        return
    nlines = []
    for line_comp, line_simp, line_ppdb, line_ppdb_rule, line_meta, line_struc in zip(
            lines_comp, lines_simp, lines_ppdb, lines_ppdb_rule, lines_meta, lines_struc):
        try:
            line_comp = line_comp.strip()
            line_simp = line_simp.strip()
            ppdb_score = float(line_ppdb)
            line_ppdb_rule = line_ppdb_rule.strip()
            # Process line of ppdb_rules
            npairs = []
            for pair in line_ppdb_rule.split('\t'):
                items = pair.split('=>')
                if len(items) <= 1:
                    continue
                if ',' in items[1] or '.' in items[1]:
                    continue
                npairs.append(pair)
            line_ppdb_rule = '\t'.join(npairs)

            line_meta = line_meta.strip()
            struc_scores = [float(v) for v in line_struc.strip().split('\t')]
            dsim = 1 - (sentence_bleu([line_simp], line_comp) + sentence_bleu([line_comp], line_simp))/2

            if is_trans and not _validate(line_comp, line_simp, ppdb_score, line_ppdb_rule, line_meta, dsim):
                continue

            # Re tokenized through Spacy
            # doc_comp, doc_simp = nlp(line_comp), nlp(line_simp)
            doc_comp, doc_simp = line_comp.split(), line_simp.split()
            line_comp = ' '.join([w for w in doc_comp]).strip().lower()
            line_simp = ' '.join([w for w in doc_simp]).strip().lower()

            feature = {
                'line_comp': line_comp.strip(),
                'line_simp': line_simp.strip(),
                'ppdb_score': ppdb_score,
                'dsim_score': dsim,
                'add_score':  struc_scores[0],
                'len_score': struc_scores[1],
                'ppdb_rule': line_ppdb_rule
            }
            nlines.append(json.dumps(feature))
            cnt += 1
        except Exception as e:
            print('excep:')
            print(e)
            traceback.print_exc()
    writer.write('\n'.join(nlines))
    print('Finished shard id %s with size %s' % (shard_id, cnt))
    time_span = datetime.now() - s_time
    print('Done id:%s with time span %s' % (shard_id, time_span))


def write_freq():
    def update_counter(path, c):
        for line in open(path):
            words = line.split()
            c.update(words)
        return c

    c = Counter()
    paths = [
        '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/ncomp/shard',
        '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/nsimp/shard',
        '/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/ner/ncomp/shard',
        '/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/ner/nsimp/shard']
    for shard_id in range(1280):
        for path in paths:
            if exists(path + str(shard_id)):
                cur_path = path + str(shard_id)
                update_counter(cur_path, c)
    vocab = []
    for w, c in c.most_common():
        vocab.append('%s\t%s' % (w, c))
    open(PATH_VOCAB_FREQ, 'w').write('\n'.join(vocab))
    print('Populate vocab.')


def read_freq():
    global global_vocab
    global_vocab = {}
    for line in open(PATH_VOCAB_FREQ):
        items = line.split('\t')
        word = items[0]
        cnt = float(items[1])
        global_vocab[word] = math.log(cnt)

if __name__ == '__main__':
    if not exists(PATH_VOCAB_FREQ) and is_trans:
        write_freq()
    read_freq()
    p = Pool(10)
    p.map(generate_example, range(1280))