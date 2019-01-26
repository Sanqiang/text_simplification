"""Generate PPDB scores for training data
Note that: use python 2.x because en package is only compatible with python 2.x
"""
from ppdb_util import populate_ppdb
from datetime import datetime
from copy import deepcopy
from multiprocessing import Pool
from os.path import exists
from os import mkdir

is_trans = False

if is_trans:
    PATH_PREFIX = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/'
    NPATH_PPDB_PREFIX = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/ppdb/'
    NPATH_PPDB_RULE_PREFIX = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/ppdb_rule/'
    NPATH_META = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/meta/'
else:
    PATH_PREFIX = '/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/raw/'
    NPATH_PPDB_PREFIX = '/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/raw/ppdb/'
    NPATH_PPDB_RULE_PREFIX = '/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/raw/ppdb_rule/'
    NPATH_META = '/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/raw/meta/'
print(PATH_PREFIX)


def sequence_contain_(seq, targets):
    if len(targets) == 0:
        print('%s_%s' % (seq, targets))
        return False
    if len(targets) > len(seq):
        return False
    for s_i, s in enumerate(seq):
        t_i = 0
        s_loop = s_i
        if s == targets[t_i]:
            while t_i < len(targets) and s_loop < len(seq) and seq[s_loop] == targets[t_i]:
                t_i += 1
                s_loop += 1
            if t_i == len(targets):
                return s_loop
    return -1


def get_best_targets_(oriwords, line_dst, widp, line_src, ignore_decay=False):
    for tar_words, weight in mapper[oriwords]:
        pos = sequence_contain_(line_dst, tar_words.split())
        if ignore_decay:
            weight_decay = 0.0
        else:
            weight_decay = max(abs(float(pos)/(len(line_dst)+0.0001)-widp)-0.2, 0.0)
        if pos != -1 and sequence_contain_(line_src, tar_words.split()) == -1:
            score = weight-weight_decay
            if score:
                return ('%s=>%s=>%s' % (oriwords, tar_words, score), score)
            else:
                return None


def get_score(line_src, line_dst, mapper):
    rule = []
    score = 0
    line_src = line_src.split()
    line_dst = line_dst.split()
    for wid in range(len(line_src)):
        # For unigram
        unigram = line_src[wid]
        if unigram in mapper and unigram not in line_dst:
            res = get_best_targets_(unigram, line_dst, float(wid)/len(line_src), line_src)
            if res:
                rule.append(res[0])
                score += res[1]

        # For bigram
        if wid + 1 < len(line_src):
            bigram = line_src[wid] + ' ' + line_src[wid + 1]
            if bigram in mapper and sequence_contain_(line_dst, (line_src[wid], line_src[wid+1])) == -1:
                res = get_best_targets_(bigram, line_dst, wid/len(line_src), line_src, ignore_decay=True)
                if res:
                    rule.append(res[0])
                    score += res[1]

        # For trigram
        if wid + 2 < len(line_src):
            trigram = line_src[wid] + ' ' + line_src[wid + 1] + ' ' + line_src[wid + 2]
            if trigram in mapper and sequence_contain_(line_dst, (line_src[wid], line_src[wid+1], line_src[wid+2])) == -1:
                res = get_best_targets_(trigram, line_dst, wid/len(line_src), line_src, ignore_decay=True)
                if res:
                    rule.append(res[0])
                    score += res[1]
    return score, '\t'.join(rule)


def get_meta(line_src, line_dst, mapper):
    """Generate meta data for is_equal |  """
    is_equal = line_src.strip() == line_dst.strip()

    set_src = set(line_src.split())
    set_dst = set(line_dst.split())
    set_extra_src = set_src - set_dst
    set_extra_dst = set_dst - set_src
    set_extra_src_cp = deepcopy(set_extra_src)
    set_extra_dst_cp = deepcopy(set_extra_dst)

    for word_src in set_extra_src_cp:
        for word_dst in set_extra_dst_cp:
            if word_src in mapper and get_best_targets_(word_src, line_dst, 0, line_src, True):
                if word_src in set_extra_src:
                    set_extra_src.remove(word_src)
                if word_dst in set_extra_dst:
                    set_extra_dst.remove(word_dst)
    meta = [
        str(is_equal),
        str(len(set_extra_src)), ' '.join(set_extra_src),
        str(len(set_extra_dst)), ' '.join(set_extra_dst)]
    return '\t'.join(meta)


def populate_file(path_comp, path_simp):
    """Populate output files based on path_comp and path_simp."""
    if not exists(path_simp) or not exists(path_comp):
        return
    f_comp = open(path_comp)
    f_simp = open(path_simp)
    lines_comp = f_comp.readlines()
    lines_simp = f_simp.readlines()
    scores = []
    rules = []
    metas = []

    ids = range(len(lines_comp))
    assert abs(len(lines_comp) - len(lines_simp)) <= 1, \
        'incorrect lines for file %s with lines %s, %s with lines %s and ids %s' % (path_comp, str(len(lines_comp)), path_simp, str(len(lines_simp)), str(len(ids)))
    for id, line_comp, line_simp in zip(ids, lines_comp, lines_simp):
        s_time = datetime.now()
        line_comp = line_comp.strip().lower()
        line_simp = line_simp.strip().lower()
        score, rule = get_score(line_comp, line_simp, mapper)
        scores.append(str(score))
        rules.append(rule)
        metas.append(get_meta(line_comp, line_simp, mapper))
        time_span = datetime.now() - s_time

    return scores, rules, metas


mapper = populate_ppdb(add_xu=False)


def generate_score_trans(id):

    path_ppdb = NPATH_PPDB_PREFIX + 'shard%s' % id
    path_ppdb_rule = NPATH_PPDB_RULE_PREFIX + 'shard%s' % id
    path_meta = NPATH_META + 'shard%s' % id
    if exists(path_ppdb) and exists(path_ppdb_rule) and exists(path_meta):
        return

    path_comp = PATH_PREFIX + '/ncomp/shard%s' % id
    path_simp = PATH_PREFIX + '/nsimp/shard%s' % id
    if not exists(path_comp) or not exists(path_simp):
        return
    print('Start id:%s' % id)

    f_ppdb = open(path_ppdb, 'w')
    f_ppdb_rule = open(path_ppdb_rule, 'w')
    f_meta = open(path_meta, 'w')

    s_time = datetime.now()

    scores, rules, metas = populate_file(path_comp, path_simp)
    f_ppdb.write('\n'.join(scores))
    f_ppdb_rule.write('\n'.join(rules))
    f_meta.write('\n'.join(metas))
    time_span = datetime.now() - s_time

    print('Done id:%s with time span %s' % (id, time_span))


if __name__ == '__main__':
    # generate_score_wikilarge()
    p = Pool(8)
    p.map(generate_score_trans, range(1280))
    # generate_score_trans(0)
