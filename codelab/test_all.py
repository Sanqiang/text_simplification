# For fix slurm cannot load PYTHONPATH
import sys
sys.path.insert(0, '/ihome/hdaqing/saz31/sanqiang/text_simplification')

import os
from model.test import test
from model.model_config import SubTestWikiEightRefConfig, SubTestWikiEightRefConfigV2, SubTestWikiEightRefConfigV2Sing
from util.arguments import get_args


args = get_args()


if __name__ == '__main__':
    mapper = {}
    path = '/zfs1/hdaqing/saz31/text_simplification/' + args.output_folder #'/Users/zhaosanqiang916/git/acl' #'/zfs1/hdaqing/saz31/text_simplification/'
    for root, dirs, files in os.walk(path):
        for file in files:
            if 'model' in root and file.endswith('.index'):
                sid = file.index('ckpt-') + len('ckpt-')
                eid = file.rindex('.index')
                step = file[sid:eid]
                resultpath = root + '/../result/eightref_test/joshua_target_' + step + '.txt'
                if not os.path.exists(resultpath):
                    ckpt = root + '/' + file[:-len('.index')]
                    test(SubTestWikiEightRefConfig(), ckpt)
                    test(SubTestWikiEightRefConfigV2(), ckpt)
                    test(SubTestWikiEightRefConfigV2Sing(), ckpt)

