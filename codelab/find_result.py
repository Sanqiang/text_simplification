import os
import operator

mapper = {}
path = '/zfs1/hdaqing/saz31/text_simplification/' #'/Users/zhaosanqiang916/git/acl' #'/zfs1/hdaqing/saz31/text_simplification/'
for root, dirs, files in os.walk(path):
    for file in files:
        if '-sari' in file and file.endswith('result') and 'result2' not in root:
            sari_sidx = file.index('-sari')
            sari_eidx = file.rindex('-', sari_sidx)
            sari = float(file[sari_sidx+len('-sari'):sari_eidx])
            mapper[root + '/' + file] = sari
mapper = sorted(mapper.items(), key=operator.itemgetter(1), reverse=True)
cnt = 10
for k,v in mapper:
    if cnt == 0:
        break
    cnt -= 1
    print(k)
    print(v)
