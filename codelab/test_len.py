from collections import Counter

c = Counter()
for line in open('/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilargenew/train/dst.txt'):
# for line in open('/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src'):
    cnt = len(line.split())
    c.update([str(cnt)])

mm = 0
for cnt in c:
    if float(cnt) > mm:
        mm = float(cnt)

print(mm)