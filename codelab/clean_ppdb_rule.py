import operator

f = open('/Users/zhaosanqiang916/git/text_simplification_data/test/wiki.full.aner.test.src.rules')
fn = open('/Users/zhaosanqiang916/git/text_simplification_data/test/wiki.full.aner.test.src.sorted.rules', 'w')
nlines = []
maxn = -1
maxl =  ''
for line in f:
    rules = line.split('\t')
    nrules = {}
    for r in rules:
        its = r.split('=>')
        if len(its) != 4:
            continue
        nr = its[1] + '=>' + its[2]
        score = float(its[3])
        if nr not in nrules:
            nrules[nr] = score
        else:
            if score > nrules[nr]:
                nrules[nr] = score
    nrules = sorted(nrules.items(), key=operator.itemgetter(1), reverse=True)

    nline = '\t'.join(['X=>' + x[0] + '=>' + str(x[1]) for x in nrules])
    nlines.append(nline)

    if len(nrules) > maxn:
        maxn = len(nrules)
        maxl = nrules

fn.write('\n'.join(nlines))
print(maxn)
print(maxl)