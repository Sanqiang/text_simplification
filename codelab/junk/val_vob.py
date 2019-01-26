import io


path_emb = '/Users/sanqiangzhao/git/ts/text_simplification_data/tmp/glove.840B.300d.txt'
path_voc = '/Users/sanqiangzhao/git/ts/text_simplification_data/tmp/comp.vocab'

f_emb = io.open(path_emb, 'r', encoding='utf-8', newline='\n', errors='ignore')
set_emb = set()
for line in f_emb:
    tokens = line.rstrip().split(' ')
    set_emb.add(tokens[0].lower())

cnt = 0.0
tcnt = 0.0
concnt = 0
set_voc = set()
for line in open(path_voc):
    token = line.rstrip().split()[0]
    if token not in set_emb:
        cnt += 1.0
        print(token)
    else:
        concnt += 1

    tcnt += 1.0

print(cnt / tcnt)
print(concnt)