import os
import operator

mapper = {}
path = '/zfs1/hdaqing/saz31/text_simplification_0424/' #'/Users/zhaosanqiang916/git/acl' #'/zfs1/hdaqing/saz31/text_simplification/'
for root, dirs, files in os.walk(path):
    if root.endswith('result'):
        print(root)