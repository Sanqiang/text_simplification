import os
import operator

mapper = {}
path = '/zfs1/hdaqing/saz31/text_simplification_0805/' #'/Users/zhaosanqiang916/git/acl' #'/zfs1/hdaqing/saz31/text_simplification/'
for root, dirs, files in os.walk(path):
    for file in files:
        if '-sari' in file and file.endswith('result') and 'eightref_val' not in root:
            sari_sidx = file.index('-sari')
            sari_eidx = file.rindex('-fkgl', sari_sidx)
            try:
                sari = float(file[sari_sidx+len('-sari'):sari_eidx])
            except:
                print('error:%s%s' % (root, file))
                sari = 0.0
            mapper[root + '/' + file] = sari
mapper = sorted(mapper.items(), key=operator.itemgetter(1), reverse=True)
cnt = 10
for k,v in mapper:
    if cnt == 0:
        break
    cnt -= 1
    print(k)
    print(v)
