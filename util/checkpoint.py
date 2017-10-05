from os import listdir
from os.path import isfile, join, exists
from os import remove, makedirs
from shutil import copy2

from model.model_config import DefaultConfig


ckpt_prefix = 'model.ckpt-'

def find_train_ckpt(path, is_outdir):
    steps = [int(f[len(ckpt_prefix):-5]) for f in listdir(path)
             if f[:len(ckpt_prefix)] == ckpt_prefix and f[-5:] == '.meta']
    if len(steps) == 0:
        if is_outdir:
            raise FileNotFoundError('No Available ckpt.')
        else:
            return None, -1
    max_step = max(steps)
    if len(steps) > 5 and is_outdir:
        del_model_files = get_model_files(sorted(steps)[:-5], path)
        for del_model_file in del_model_files:
            remove(path + del_model_file)

    model_files = get_model_files(max_step, path)
    return model_files, max_step

def get_model_files(steps, path):
    if not isinstance(steps, list):
        steps = [steps]
    model_files = []
    for step in steps:
        model_pref = ckpt_prefix + str(step)
        model_files.extend([f for f in listdir(path)
                            if isfile(join(path, f)) and f[:len(model_pref)] == model_pref])
    return model_files


def copy_ckpt_to_modeldir(modeldir, outdir):
    if not exists(modeldir):
        makedirs(modeldir)
    if not exists(outdir):
        makedirs(outdir)

    files, max_step = find_train_ckpt(outdir, True)
    _, cur_max_step = find_train_ckpt(modeldir, False)
    if cur_max_step == max_step:
        raise FileNotFoundError('No new ckpt.')

    for file in files:
        source = outdir + file
        target = modeldir + file
        copy2(source, target)
        print('Copy Ckpt from %s \t to \t %s.' % (source, target))
    return modeldir + ckpt_prefix + str(max_step)



if __name__ == '__main__':
    ckpt = copy_ckpt_to_modeldir(DefaultConfig())
    print(ckpt)
