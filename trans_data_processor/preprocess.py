"""Deprecated: because we use stanford NLP for ner
Multi process for processing the trans data"""
import spacy
from collections import defaultdict
from datetime import datetime
from os.path import exists
from os import mkdir
from multiprocessing import Pool


nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger'])

PATH_PREFIX = '/zfs1/hdaqing/saz31/dataset/tmp_trans'
NPATH_PREFIX = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner'

# PATH_PREFIX = '/home/zhaos5/tmp/'
# NPATH_PREFIX = '/home/zhaos5/tmp/ner/'

def preprocess_line(line_comp, line_simp, is_test=False):
    """Generate ner mapper and tokenized string."""
    def _is_asc2(sent):
        return all(ord(char) < 128 for char in sent)

    def post_replace(sent):
        sent = sent.replace('``', '"')
        sent = sent.replace('\'\'', '"')
        sent = sent.replace('`', '\'')
        return sent

    if not is_test and (not _is_asc2(line_comp) or not _is_asc2(line_simp)):
        raise ValueError('Non ASC2 Sentence.')
    if len(line_comp.split()) >= 100 or len(line_simp.split()) >= 100:
        raise ValueError('Too Long Sentence.')

    doc_comp, doc_simp = nlp(line_comp), nlp(line_simp)

    mapper_comp = {}
    label_counter_comp = defaultdict(int)
    for ent in doc_comp.ents:
        ent_label = ent.label_ + str(label_counter_comp[ent.label_])
        line_comp = line_comp.replace(ent.text, ent_label)
        mapper_comp[ent_label] = ent.text
        label_counter_comp[ent.label_] += 1

    mapper_simp = {}
    label_counter_simp = defaultdict(int)
    for ent in doc_simp.ents:
        ent_label = ent.label_ + str(label_counter_simp[ent.label_])
        line_simp = line_simp.replace(ent.text, ent_label)
        mapper_simp[ent_label] = ent.text
        label_counter_simp[ent.label_] += 1

    doc_comp, doc_simp = nlp(line_comp), nlp(line_simp)
    words_comp = ' '.join([w.text for w in doc_comp]).strip().lower()
    words_simp = ' '.join([w.text for w in doc_simp]).strip().lower()
    words_comp = post_replace(words_comp)
    words_simp = post_replace(words_simp)

    mapper_comp_str = '\t'.join(['%s=>%s' % (mapper_comp[ent_label], ent_label)
                       for ent_label in mapper_comp])
    mapper_simp_str = '\t'.join(['%s=>%s' % (mapper_simp[ent_label], ent_label)
                       for ent_label in mapper_simp])

    return words_comp, words_simp, mapper_comp_str, mapper_simp_str


def preprocess_file(path_comp, path_simp, is_test=False):
    """Generate ner mapper and tokenized str for path."""
    s_time = datetime.now()
    words_comps, words_simps, mapper_comp_strs, mapper_simp_strs = [], [], [], []
    lines_comp = open(path_comp).readlines()
    lines_simp = open(path_simp).readlines()
    for line_comp, line_simp in zip(lines_comp, lines_simp):
        try:
            words_comp, words_simp, mapper_comp_str, mapper_simp_str = preprocess_line(line_comp, line_simp, is_test)
            words_comps.append(words_comp)
            words_simps.append(words_simp)
            mapper_comp_strs.append(mapper_comp_str)
            mapper_simp_strs.append(mapper_simp_str)
        except:
            pass

        if len(words_comps) % 100 == 0:
            time_span = datetime.now() - s_time
            print('Done line %s with time span %s' % (len(words_comps), time_span))
            s_time = datetime.now()

    return words_comps, words_simps, mapper_comp_strs, mapper_simp_strs


def process_trans(id):
    npaths = [NPATH_PREFIX + '/ncomp/', NPATH_PREFIX + '/nsimp/',
              NPATH_PREFIX + '/ncomp_map/', NPATH_PREFIX + '/nsimp_map/']
    for npath in npaths:
        if not exists(npath):
            mkdir(npath)

    if not exists(PATH_PREFIX + '/ncomp/shard%s' % id) or not exists(PATH_PREFIX + '/nsimp/shard%s' % id):
        return

    if (exists(NPATH_PREFIX + '/ncomp/shard%s' % id) and
        exists(NPATH_PREFIX + '/nsimp/shard%s' % id) and
        exists(NPATH_PREFIX + '/ncomp_map/shard%s' % id) and
        exists(NPATH_PREFIX + '/nsimp_map/shard%s' % id)):
        return

    f_ncomp = open(NPATH_PREFIX + '/ncomp/shard%s' % id, 'w')
    f_nsimp = open(NPATH_PREFIX + '/nsimp/shard%s' % id, 'w')
    f_ncomp_map = open(NPATH_PREFIX + '/ncomp_map/shard%s' % id, 'w')
    f_nsimp_map = open(NPATH_PREFIX + '/nsimp_map/shard%s' % id, 'w')

    words_comps, words_simps, mapper_comp_strs, mapper_simp_strs = preprocess_file(
        PATH_PREFIX + '/ncomp/shard%s' % id, PATH_PREFIX + '/nsimp/shard%s' % id)

    print('Start id:%s' % id)
    s_time = datetime.now()

    f_ncomp.write('\n'.join(words_comps))
    f_nsimp.write('\n'.join(words_simps))
    f_ncomp_map.write('\n'.join(mapper_comp_strs))
    f_nsimp_map.write('\n'.join(mapper_simp_strs))
    time_span = datetime.now() - s_time
    print('Done id:%s with time span %s' % (id, time_span))


def generate_score_wikilarge():
    # for train
    path_comp = '/Users/sanqiangzhao/git/ts/text_simplification_data/train/wikilarge/wiki.full.aner.ori.train.src'
    path_simp = '/Users/sanqiangzhao/git/ts/text_simplification_data/train/wikilarge/wiki.full.aner.ori.train.dst'
    npath_base = '/Users/sanqiangzhao/git/ts/text_simplification_data/train/wikilarge/'
    words_comps, words_simps, mapper_comp_strs, _ = preprocess_file(
        path_comp, path_simp, is_test=False)
    open(npath_base + '/words_comps', 'w').write('\n'.join(words_comps))
    open(npath_base + '/words_simps', 'w').write('\n'.join(words_simps))
    open(npath_base + '/mapper_comp_strs', 'w').write('\n'.join(mapper_comp_strs))


    # For eval
    # path_comp = '/Users/sanqiangzhao/git/ts/text_simplification_data/val_0930/norm.ori'
    # npath_base = '/Users/sanqiangzhao/git/ts/text_simplification_data/val_0930/'
    #
    # words_comps, _, mapper_comp_strs, _ = preprocess_file(
    #     path_comp, path_comp, is_test=True)
    # open(npath_base + '/words_comps', 'w').write('\n'.join(words_comps))
    # open(npath_base + '/mapper_comp_strs', 'w').write('\n'.join(mapper_comp_strs))



if __name__ == '__main__':
    # p = Pool(4)
    # p.map(process_trans, range(1280))
    generate_score_wikilarge()
