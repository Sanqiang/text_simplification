base = '/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilargenew/trainx/'
rules0 = open(base + 'rule_mappern.txt').readlines()
rules1 = open(base + 'rule_mappery.txt').readlines()

for lid in range(len(rules0)):
    rule0 = rules0[lid].strip()
    rule1 = rules1[lid].strip()
    rule0 = set(['=>'.join(r.split('=>')[1:4]) for r in rule0.split('\t')])
    rule1 = set(['=>'.join(r.split('=>')[1:4]) for r in rule1.split('\t')])
    diff = rule1 - rule0
    if diff:
        print('%s\t%s' % (lid, diff))


