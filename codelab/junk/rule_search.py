import os

comp_str = 'never record'
simp_str = 'unknown'

PATH = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/'
PATH_COMP = PATH + 'ncomp/shard'
PATH_SIMP = PATH + 'nsimp/shard'

for shard_i in range(1280):

    path_comp = PATH_COMP + str(shard_i)
    path_simp = PATH_SIMP + str(shard_i)
    if not os.path.exists(path_comp) or not os.path.exists(path_simp):
        continue

    lines_comp = open(path_comp).readlines()
    lines_simp = open(path_simp).readlines()

    assert len(lines_comp) == len(lines_simp)

    for line_comp, line_simp in zip(lines_comp, lines_simp):
        if comp_str in line_comp and simp_str in line_simp:
            print(line_comp)
            print(line_simp)
            print('=========')
