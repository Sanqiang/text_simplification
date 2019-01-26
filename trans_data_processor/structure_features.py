"""Generate structure scores for training data
"""
import spacy
import sys
from multiprocessing import Pool
from os.path import exists
from datetime import datetime


nlp = spacy.load('en_core_web_lg', disable=['tagger', 'ner'])

is_trans = False

if is_trans:
    PATH_PREFIX = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/'
    NPATH_STRUC_PREFIX = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/struc/'
else:
    PATH_PREFIX = '/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/raw/'
    NPATH_STRUC_PREFIX = '/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/raw/struc/'

def get_score(line_comp, line_simp):
    """Get score for
        [add_score:http://www.paaljapan.org/conference2011/ProcNewest2011/pdf/poster/P-13.pdf],
        length of simple sentence,
        add_score for complex sentence
        length of complex sentence
        ratio of length of simple sentence over complex sentence
    """
    len_simp = len(line_simp.split())
    len_comp = len(line_comp.split())
    ratio = float(len_simp) / (1e-12 + float(len_comp))

    if sys.version_info < (3, 0):
        line_simp = line_simp.decode('utf8')

    doc_simp = nlp(line_simp)
    dist_simp = 0.0
    cnt_simp = 0.0
    for wd in doc_simp:
        dist_simp += abs(wd.head.i-wd.i)
        cnt_simp += 1
    add_score_simp = float(dist_simp) / (1e-12 + cnt_simp)

    if sys.version_info < (3, 0):
        line_comp = line_comp.decode('utf8')

    doc_comp = nlp(line_comp)
    dist_comp = 0.0
    cnt_comp = 0.0
    for wd in doc_comp:
        dist_comp += abs(wd.head.i - wd.i)
        cnt_comp += 1
    add_score_comp = float(dist_comp) / (1e-12 + cnt_comp)

    return '\t'.join([str(add_score_simp), str(len_simp),
                      str(add_score_comp), str(len_comp),
                      str(ratio)])


def populate_file(path_comp, path_simp):
    """Populate output files based on path_comp and path_simp."""
    if not exists(path_simp) or not exists(path_comp):
        print('not existed for %s,%s' % (path_comp, path_simp))
        return
    f_comp = open(path_comp)
    f_simp = open(path_simp)
    lines_comp = f_comp.readlines()
    lines_simp = f_simp.readlines()
    scores = []
    for line_comp, line_simp in zip(lines_comp, lines_simp):
        line_comp = line_comp.strip().lower()
        line_simp = line_simp.strip().lower()
        score = get_score(line_comp, line_simp)
        scores.append(str(score))
    return scores


def generate_score_trans(id):
    path_struc = NPATH_STRUC_PREFIX + 'shard%s' % id
    if exists(path_struc):
        return
    # print('Start id:%s' % id)
    path_comp = PATH_PREFIX + '/ncomp/shard%s' % id
    path_simp = PATH_PREFIX + '/nsimp/shard%s' % id
    if not exists(path_comp) or not exists(path_simp):
        return

    f_struc = open(path_struc, 'w')

    s_time = datetime.now()

    scores = populate_file(path_comp, path_simp)
    f_struc.write('\n'.join(scores))
    time_span = datetime.now() - s_time

    print('Done id:%s with time span %s' % (id, time_span))
    f_struc.close()

# Deprecated: move wikilarge process into multiprocessor so only change the path
# def generate_score_wikilarge():
#     path_comp = '/Users/sanqiangzhao/git/ts/text_simplification_data/train/wikilarge/wiki.full.aner.ori.train.src'
#     path_simp = '/Users/sanqiangzhao/git/ts/text_simplification_data/train/wikilarge/wiki.full.aner.ori.train.dst'
#     npath_base = '/Users/sanqiangzhao/git/ts/text_simplification_data/train/wikilarge/'
#     scores = populate_file(path_comp, path_simp)
#     path_struc = npath_base + 'struc'
#     open(path_struc, 'w').write('\n'.join(scores))


if __name__ == '__main__':
    # generate_score_wikilarge()
    p = Pool(10)
    p.map(generate_score_trans, range(1280))
    # generate_score_trans(0)
