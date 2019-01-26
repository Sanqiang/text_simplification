"""Util of PPDB
Note that: use python 2.x because en package is only compatible with python 2.x
"""
import re
import en
from datetime import datetime

def verb_ops(nori_words, tar_words, mapper, weight, checker, ing=False):
    # Raw
    try:
        ntar_words = tar_words.split()
        ntar_words[0] = ntar_words[0]
        ntar_words = ' '.join(ntar_words)
        if nori_words + '=>' + ntar_words not in checker:
            checker.add(nori_words + '=>' + ntar_words)
            if nori_words not in mapper:
                mapper[nori_words] = []
            if valid_mapper_insert(nori_words, ntar_words, mapper):
                mapper[nori_words].append((ntar_words, weight))
    except:
        pass

    # Past
    try:
        ntar_words = tar_words.split()
        ntar_words[0] = en.verb.past(ntar_words[0])
        ntar_words = ' '.join(ntar_words)
        if nori_words + '=>' + ntar_words not in checker:
            checker.add(nori_words + '=>' + ntar_words)
            if nori_words not in mapper:
                mapper[nori_words] = []
            if ntar_words and valid_mapper_insert(nori_words, ntar_words, mapper):
                mapper[nori_words].append((ntar_words, weight))
    except:
        pass

    # present 1/2/3
    try:
        ntar_words = tar_words.split()
        ntar_words[0] = en.verb.present(ntar_words[0], person=1)
        ntar_words = ' '.join(ntar_words)
        if nori_words + '=>' + ntar_words not in checker:
            checker.add(nori_words + '=>' + ntar_words)
            if nori_words not in mapper:
                mapper[nori_words] = []
            if ntar_words and valid_mapper_insert(nori_words, ntar_words, mapper):
                mapper[nori_words].append((ntar_words, weight))
    except:
        pass
    try:
        ntar_words = tar_words.split()
        ntar_words[0] = en.verb.present(ntar_words[0], person=2)
        ntar_words = ' '.join(ntar_words)
        if nori_words + '=>' + ntar_words not in checker:
            checker.add(nori_words + '=>' + ntar_words)
            if nori_words not in mapper:
                mapper[nori_words] = []
            if ntar_words and valid_mapper_insert(nori_words, ntar_words, mapper):
                mapper[nori_words].append((ntar_words, weight))
    except:
        pass
    try:
        ntar_words = tar_words.split()
        ntar_words[0] = en.verb.present(ntar_words[0], person=3)
        ntar_words = ' '.join(ntar_words)
        if nori_words + '=>' + ntar_words not in checker:
            checker.add(nori_words + '=>' + ntar_words)
            if nori_words not in mapper:
                mapper[nori_words] = []
            if ntar_words and valid_mapper_insert(nori_words, ntar_words, mapper):
                mapper[nori_words].append((ntar_words, weight))
    except:
        pass
    if ing:
        # present_participle/past_participle
        try:
            ntar_words = tar_words.split()
            ntar_words[0] = en.verb.present_participle(ntar_words[0])
            ntar_words = ' '.join(ntar_words)
            if nori_words + '=>' + ntar_words not in checker:
                checker.add(nori_words + '=>' + ntar_words)
                if nori_words not in mapper:
                    mapper[nori_words] = []
                if ntar_words and valid_mapper_insert(nori_words, ntar_words, mapper):
                    mapper[nori_words].append((ntar_words, weight))
        except:
            pass
    try:
        ntar_words = tar_words.split()
        ntar_words[0] = en.verb.past_participle(ntar_words[0])
        ntar_words = ' '.join(ntar_words)
        if nori_words + '=>' + ntar_words not in checker:
            checker.add(nori_words + '=>' + ntar_words)
            if nori_words not in mapper:
                mapper[nori_words] = []
            if ntar_words and valid_mapper_insert(nori_words, ntar_words, mapper):
                mapper[nori_words].append((ntar_words, weight))
    except:
            pass

    return mapper


def verb_ops_ori(ori_words, tar_words, mapper, weight, checker):
    # Raw
    try:
        nori_words = ori_words.split()
        nori_words[0] = nori_words[0]
        nori_words = ' '.join(nori_words)
        if nori_words not in checker:
            checker.add(nori_words)
            mapper = verb_ops(nori_words, tar_words, mapper, weight, checker)
    except:
        pass

    #Past
    try:
        nori_words = ori_words.split()
        nori_words[0] = en.verb.past(nori_words[0])
        nori_words = ' '.join(nori_words)
        if nori_words not in checker:
            checker.add(nori_words)
            mapper = verb_ops(nori_words, tar_words, mapper, weight, checker)
    except:
        pass

    # present 1/2/3
    try:
        nori_words = ori_words.split()
        nori_words[0] = en.verb.present(nori_words[0], person=1)
        nori_words = ' '.join(nori_words)
        if nori_words not in checker:
            checker.add(nori_words)
            mapper = verb_ops(nori_words, tar_words, mapper, weight, checker)
    except:
        pass
    try:
        nori_words = ori_words.split()
        nori_words[0] = en.verb.present(nori_words[0], person=2)
        nori_words = ' '.join(nori_words)
        if nori_words not in checker:
            checker.add(nori_words)
            mapper = verb_ops(nori_words, tar_words, mapper, weight, checker)
    except:
        pass
    try:
        nori_words = ori_words.split()
        nori_words[0] = en.verb.present(nori_words[0], person=3)
        nori_words = ' '.join(nori_words)
        if nori_words not in checker:
            checker.add(nori_words)
            mapper = verb_ops(nori_words, tar_words, mapper, weight, checker)
    except:
        pass

    # present_participle/past_participle
    try:
        nori_words = ori_words.split()
        nori_words[0] = en.verb.present_participle(nori_words[0])
        nori_words = ' '.join(nori_words)
        if nori_words not in checker:
            checker.add(nori_words)
            mapper = verb_ops(nori_words, tar_words, mapper, weight, checker, ing=True)
    except:
        pass
    try:
        nori_words = ori_words.split()
        nori_words[0] = en.verb.past_participle(nori_words[0])
        nori_words = ' '.join(nori_words)
        if nori_words not in checker:
            checker.add(nori_words)
            mapper = verb_ops(nori_words, tar_words, mapper, weight, checker)
    except:
        pass
    return mapper


def is_alpha_srt(word):
    for ch in word:
        if (ch < 'a' or ch > 'z') and ch != ' ':
            return False
    return True


def lcs(a, b):
    lengths = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    # read the substring out from the matrix
    result = ""
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x - 1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y - 1]:
            y -= 1
        else:
            assert a[x - 1] == b[y - 1]
            result = a[x - 1] + result
            x -= 1
            y -= 1
    return result


def valid_mapper_insert(ori_words, tar_words, mapper):
    if tar_words not in mapper:
        return True
    for wds, _ in mapper[tar_words]:
        if wds == ori_words:
            return False
    return True


def process_line(line, mapper, is_xu=False):
    items = line.strip().lower().split('\t')
    if len(items) < 5 or items[2] == '[cd]':
        return
    ori_words = items[3]
    tar_words = items[4]

    try:
        if en.verb.infinitive(ori_words) == en.verb.infinitive(tar_words):
            return
        # elif float(len(lcs(ori_words, tar_words))) / max(len(ori_words), len(tar_words)) >= 0.7:
        #     return
    except:
        pass

    if not ori_words or not tar_words:
        return

    checker = set()
    if is_xu:
        weight = 0.1
    else:
        weight = float(items[1])
    if ori_words not in mapper:
        mapper[ori_words] = []
    if valid_mapper_insert(ori_words, tar_words, mapper):
        mapper[ori_words].append((tar_words, weight))
    if items[2].startswith('[nn'):
        try:
            nori_words = ori_words.split()
            nori_words[0] = en.noun.plural(nori_words[0])
            nori_words = ' '.join(nori_words)
            ntar_words = tar_words.split()
            ntar_words[0] = en.noun.plural(ntar_words[0])
            ntar_words = ' '.join(ntar_words)
            if nori_words not in mapper:
                mapper[nori_words] = []
            if ntar_words and valid_mapper_insert(nori_words, ntar_words, mapper):
                mapper[nori_words].append((ntar_words, weight))
        except:
            pass

    if items[2].startswith('[v'):
        mapper = verb_ops_ori(ori_words, tar_words, mapper, weight, checker)
    if 'be' in ori_words:
        nori_words = ori_words.replace('be', 'is')
        ntar_words = ori_words.replace('be', 'is')
        if nori_words not in mapper:
            mapper[nori_words] = []
        if ntar_words and valid_mapper_insert(nori_words, ntar_words, mapper):
            mapper[nori_words].append((ntar_words, weight))

        nori_words = ori_words.replace('be', 'am')
        ntar_words = ori_words.replace('be', 'am')
        if nori_words not in mapper:
            mapper[nori_words] = []
        if ntar_words and valid_mapper_insert(nori_words, ntar_words, mapper):
            mapper[nori_words].append((ntar_words, weight))

        nori_words = ori_words.replace('be', 'are')
        ntar_words = ori_words.replace('be', 'are')
        if nori_words not in mapper:
            mapper[nori_words] = []
        if ntar_words and valid_mapper_insert(nori_words, ntar_words, mapper):
            mapper[nori_words].append((ntar_words, weight))


def populate_ppdb(add_xu=True):
    print('Start Process PPDB.')
    mapper = {}
    s_t = datetime.now()
    cnt = 0
    # for line in open('/home/zhaos5/tmp/SimplePPDB.enrich'):
    # for line in open('/Users/sanqiangzhao/git/text_simplification_data/ppdb/SimplePPDB.enrich'):
    for line in open('/zfs1/hdaqing/saz31/dataset/tmp_trans/code/SimplePPDB.enrich'):
        process_line(line, mapper, False)
        cnt += 1
        if cnt % 50000 == 0:
            e_t = datetime.now()
            sp = e_t - s_t
            print('Process PPDB line\t%s use\t%s' % (str(cnt), str(sp)))
    if add_xu:
        for line in open('/zfs1/hdaqing/saz31/dataset/tmp_trans/code/XU_PPDB'):
            process_line(line, mapper, True)
            cnt += 1
            if cnt % 50000 == 0:
                e_t = datetime.now()
                sp = e_t - s_t
                print('Process PPDB line\t%s use\t%s' % (str(cnt), str(sp)))
    # for line in open('/zfs1/hdaqing/saz31/dataset/tmp_trans/code/ppdb-2.0-xxxl-lexical.processed'):
    #     process_line(line, mapper)
    #     cnt += 1
    #     if cnt % 50000 == 0:
    #         e_t = datetime.now()
    #         sp = e_t - s_t
    #         print('Process PPDB line\t%s use\t%s' % (str(cnt), str(sp)))
    return mapper

# mapper = populate_ppdb()

