import json
import os


PATH_FEATURES = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/features/shard'


def print_valid(id):
    if not os.path.exists(PATH_FEATURES + str(id)):
        return
    for line in open(PATH_FEATURES + str(id)):
        feature = json.loads(line)
        line_comp = feature['line_comp']
        line_simp = feature['line_simp']
        if 'neptune' in line_simp:
            print(line_comp)
            print(line_simp)
            print('======')


for i in range(1280):
    print_valid(i)