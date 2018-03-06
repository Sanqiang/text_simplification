from os import listdir
from os.path import isfile, join, exists
from os import remove, makedirs
from shutil import copy2

from model.model_config import DefaultConfig


ckpt_prefix = 'model.ckpt-'

def find_train_ckptfiles(path, is_delete):
    """Find checkpoint files based on its max steps.
       is_outdir indicates whether find from outdir or modeldir.
       note that outdir generated from train and eval copy them to modeldir.
    """
    steps = [int(f[len(ckpt_prefix):-5]) for f in listdir(path)
             if f[:len(ckpt_prefix)] == ckpt_prefix and f[-5:] == '.meta']
    if len(steps) == 0:
        if is_delete:
            raise FileNotFoundError('No Available ckpt.')
        else:
            return None, -1
    max_step = max(steps)
    if len(steps) > 5 and is_delete:
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


def copy_ckpt_to_modeldir(modeldir, logdir):
    if not exists(modeldir):
        makedirs(modeldir)
    if not exists(logdir):
        makedirs(logdir)

    files, max_step = find_train_ckptfiles(logdir, False)
    _, cur_max_step = find_train_ckptfiles(modeldir, False)
    if cur_max_step == max_step:
        raise FileNotFoundError('No new ckpt. cur_max_step: %s, max_step: %s.'
                                % (cur_max_step, max_step))

    for file in files:
        source = logdir + file
        target = modeldir + file
        copy2(source, target)
        print('Copy Ckpt from %s \t to \t %s.' % (source, target))
    return modeldir + ckpt_prefix + str(max_step)


def backup_log(logdir):
    """Back up the log and remove the current log, so that he train can use init_fn()"""
    files = listdir(logdir)
    back_logdir = logdir + '../log_bk'
    if not exists(back_logdir):
        makedirs(back_logdir)
    for file in files:
        copy2(logdir + file, back_logdir)
    for file in files:
        remove(logdir + file)

def find_train_ckptpaths(outdir):
    _, step = find_train_ckptfiles(outdir, False)
    return outdir + ckpt_prefix + str(step)


if __name__ == '__main__':
    ckpt = copy_ckpt_to_modeldir(DefaultConfig())
    print(ckpt)
