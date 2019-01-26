PATH = '/Users/sanqiang/git/ts/text_simplification_data/newsela/download/newsela_articles_20150302.aligned.sents.txt'

simp_checker = {}
for line in open(PATH):
    items = line.split('\t')
    id = int(items[0][len('DOC'):])
    if id > 1070:
        # avoid get samples for valid/test
        continue
    ori_level = int(items[1][len('V'):])
    tar_level = int(items[2][len('V'):])
    if ori_level < tar_level:
        ori_sent = items[3].strip()
        tar_sent = items[4].strip()
        if ori_sent in simp_checker:
            if simp_checker[ori_sent][0] > tar_level:
                del simp_checker[ori_sent]
                simp_checker[ori_sent] = (tar_level, tar_sent)
        else:
            simp_checker[ori_sent] = (tar_level, tar_sent)


ori_sents, tar_sents = [], []
for ori_sent in simp_checker:
    tar_sent = simp_checker[ori_sent][1]
    ori_sents.append(ori_sent)
    tar_sents.append(tar_sent)

open('/Users/sanqiang/git/ts/text_simplification_data/newsela/train/ori.train.src', 'w').write('\n'.join(ori_sents))
open('/Users/sanqiang/git/ts/text_simplification_data/newsela/train/ori.train.dst', 'w').write('\n'.join(tar_sents))

