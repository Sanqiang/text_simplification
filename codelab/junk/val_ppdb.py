path_ppdb = '/Users/sanqiangzhao/git/ts/text_simplification_data/ppdb/SimplePPDB.enrich'

for line in open(path_ppdb):
    items = line.split()
    if len(items) < 5:
        continue
    ori, tar = items[3], items[4]
    if '-' in ori and '-rrb-':
        print(line)