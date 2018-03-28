from collections import Counter

c = Counter()
for line in open('/Users/zhaosanqiang916/git/text_simplification_data/train/dress/nwikilarge/train/dst.txt'):
    words = line.split()
    c.update(words)

f = open('/Users/zhaosanqiang916/git/text_simplification_data/train/dress/nwikilarge/train/voc_dst.txt', 'w')
c = c.most_common()
for wd, cnt in c:
    f.write(wd)
    f.write('\t')
    f.write(str(cnt))
    f.write('\n')
f.close()
