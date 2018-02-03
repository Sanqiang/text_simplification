nlines = []
for line in open('/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.rules.v3.sing.tml'):
    if ' ' not in line and ',' not in line and ';' not in line and '\'' not in line:
        nlines.append(line.strip())

f = open('/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.rules.v3.sing', 'w')
f.write('\n'.join(nlines))
f.close()


