# from nltk.translate.bleu_score import sentence_bleu
#
# ref = [['c1', 'c2', 's3', 'c4', 'c5', 'c6', 'x'],['c1', 's2', 'c3', 's4', 'c5', 'c6', 'x'], ['s1', 'c2', 'c3', 'c4', 'c5', 'c6']]
# out2 = ['c1','c2','c3', 'c4', 'c5', 'c6']
# bleu1 = sentence_bleu(ref, out2, weights=[1])
# print(bleu1)
#
#
# out2 = ['s1','s2','s3', 'c4', 'c5']
# bleu2 = sentence_bleu(ref, out2, weights=[1])
# print(bleu2)



# f = open('/Users/zhaosanqiang916/Google Drive/acldata/sari_report')
# nlines = []
# for line in f:
#     if '=' in line and '==' not in line:
#         name = line.split('=')[0].strip()
#         val = float(line.split('=')[1].strip())
#         nlines.append(name + ' = ' + str(100 * round(val, 4)))
#     else:
#         nlines.append(line.strip())
#
#
# f2 = open('/Users/zhaosanqiang916/Google Drive/acldata/sari_report2', 'w')
# f2.write('\n'.join(nlines))
# f2.close()

lower_file = '/Users/zhaosanqiang916/git/text_simplification_data/val/tune.8turkers.tok.simp'
process_file = '/Users/zhaosanqiang916/git/text_simplification_data/val/tune.8turkers.tok.simp.ner'

mapper_file = open('/Users/zhaosanqiang916/git/text_simplification_data/val/tune.8turkers.tok.map')
mappers = []
for line in mapper_file:
    mapper = {}
    rules = line.strip().split("\t")
    for rule in rules:
        items = rule.split('=>')
        if len(items) != 2:
            continue
        word = items[0].lower()
        ner = items[1]
        mapper[word] = ner
    mappers.append(mapper)

nlines = []
for i, line in enumerate(open(lower_file).readlines()):
    for word in mappers[i]:
        ner = mappers[i][word]
        if word in line:
            line = line.replace(word, ner)
    nlines.append(line)

open(process_file, 'w').write(''.join(nlines))

