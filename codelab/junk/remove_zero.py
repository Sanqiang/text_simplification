import os

PATH = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/struc'

files = os.listdir(PATH)

cnt = 0
for file in files:
    cur_path = os.path.join(PATH, file)
    if len(open(cur_path).readlines()) == 0:
        os.remove(cur_path)
        cnt += 1
print(cnt)
