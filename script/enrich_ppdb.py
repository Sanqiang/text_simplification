# Enrich SimplePPDB
import en

path = '/Users/zhaosanqiang916/git/text_simplification_data/ppdb/SimplePPDB'
lines = []
f = open('/Users/zhaosanqiang916/git/text_simplification_data/ppdb/SimplePPDB.enrich', 'w')

processed = 0
for line in open(path):
    line = line.strip()
    lines.append(line)
    items = line.split('\t')
    tag = items[2]
    ori_words = items[3].strip()
    tar_words = items[4].strip()
    if tag[0] == '[':
        tag = tag[1:]
    if tag[-1] == ']':
        tag = tag[:-1]

    if tag[:2] == 'VB':
        try:
            nori_word = en.verb.past_participle(ori_words)
            ntar_words = en.verb.past_participle(tar_words)
            nline = '\t'.join([items[0], items[1], '[NEW]', nori_word, ntar_words])
            lines.append(nline)
        except:
            line = ''
            # print 'Error\t%s' % line

    if len(lines) % 10000 == 0:
        f.write('\n'.join(lines))
        f.flush()
        lines = []
        processed += 1
        print 'processed\t%s.' % processed
f.write('\n'.join(lines))
f.flush()
f.close()


