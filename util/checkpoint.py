from os import listdir
from os.path import isfile, join
from shutil import copy2

from model.model_config import DefaultConfig


ckpt_prefix = 'model.ckpt-'

def find_train_ckpt(model_config):
    steps = [int(f[len(ckpt_prefix):-5]) for f in listdir(model_config.outdir)
             if f[:len(ckpt_prefix)] == ckpt_prefix and f[-5:] == '.meta']
    max_step = max(steps)

    model_pref = ckpt_prefix + str(max_step)
    model_files = [f for f in listdir(model_config.outdir)
                 if isfile(join(model_config.outdir, f)) and f[:len(model_pref)] == model_pref]
    print('Find Train Ckpt with step %d.' % max_step)

    return model_files, max_step


def copy_ckpt_to_modeldir(model_config):
    files, max_step = find_train_ckpt(model_config)
    for file in files:
        source = model_config.outdir + file
        target = model_config.modeldir + file
        copy2(source, target)
        print('Copy Ckpt from %s \t to \t %s.' % (source, target))
    return model_config.modeldir + ckpt_prefix + str(max_step)


if __name__ == '__main__':
    ckpt = copy_ckpt_to_modeldir(DefaultConfig())
    print(ckpt)
