"""Generate features for normal sentence for eval (so that we can scale based on it)
    each lines contains "add score\tlength"
"""
from trans_data_processor.structure_features import get_score as get_struc_score

PATH_COMP = '/Users/sanqiangzhao/git/ts/text_simplification_data/test2/ncomp/test.8turkers.tok.norm.ori'
PATH_FEATURES = '/Users/sanqiangzhao/git/ts/text_simplification_data/test2/ncomp/test.8turkers.tok.norm.features'


nlines = []
for line in open(PATH_COMP):
    scores = get_struc_score(line, line)
    items = scores.split('\t')
    add_score, length = items[0], items[1]
    nlines.append('%s\t%s' % (add_score, length))

f = open(PATH_FEATURES, 'w')
f.write('\n'.join(nlines))
f.close()