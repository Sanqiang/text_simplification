from collections import Counter
filterout = set()
nlines = []
base = '/Users/zhaosanqiang916/git/ts/text_simplification_data/train/wikilarge/'
for line in open(base + 'rule_mapper.txt'):
    line = line.strip().lower()
    nline = []
    for rule in line.split('\t'):
        items = rule.split('=>')
        if len(items) != 4:
            continue
        ori = items[1]
        tar = items[2]
        wei = float(items[3])
        if wei < 0.01:
            filterout.add(rule)
            continue
        nline.append(rule)
    nlines.append('\t'.join(nline))

open(base + 'rule_mapper2.txt', 'w').write('\n'.join(nlines))

c = Counter()
for line in open(base + 'rule_mapper2.txt'):
    rules = line.strip().split('\t')
    for rule in rules:
        if rule:
            weight = float(rule.split('=>')[-1])
            rule = '=>'.join(rule.split('=>')[1:3])
            if rule not in c:
                c[rule] = weight
            else:
                c[rule] = max(weight, c[rule])

rules = []
for rule, cnt in c.most_common():
    items = rule.split('=>')
    if len(items) == 2:
        rule = '%s=>%s' % (items[0], items[1])
        rules.append('%s\t%s' % (rule, str(cnt)))

f = open(base + 'rule_voc.txt', 'w')
f.write('\n'.join(rules))
f.close()