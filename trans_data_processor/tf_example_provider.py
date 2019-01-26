"""Create tensroflow.Example for model input."""
import tensorflow as tf
import json

from os.path import exists
from datetime import datetime
from multiprocessing import Pool

is_trans = False

if is_trans:
    # for trans
    PATH_TRAIN = '/zfs1/hdaqing/saz31/dataset/tf_example/ppdb/train.tfrecords.'
    PATH_PREFIX_FEATURES = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/features/shard'
else:
    # for wikilarge
    PATH_TRAIN = '/zfs1/hdaqing/saz31/dataset/tf_example/ppdb_ori/wiki.tfrecords.'
    PATH_PREFIX_FEATURES = '/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/ner/features/shard'

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_features(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _byte_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(value, 'utf-8')]))

def generate_example(shard_id):
    cnt = 0
    if not exists(PATH_PREFIX_FEATURES + str(shard_id)):
        return
    writer = tf.python_io.TFRecordWriter(PATH_TRAIN + str(shard_id))
    print('Start shard id %s' % shard_id)
    s_time = datetime.now()
    lines_features = open(PATH_PREFIX_FEATURES + str(shard_id)).readlines()

    for line_features in lines_features:
        obj = json.loads(line_features)
        line_comp = obj['line_comp']
        line_simp = obj['line_simp']
        line_ppdbrule = obj['ppdb_rule']
        ppdb_score = float(obj['ppdb_score'])
        dsim_score = float(obj['dsim_score'])
        add_score = float(obj['add_score'])
        len_score = float(obj['len_score'])

        feature = {
            'line_comp': _byte_features(line_comp),
            'line_simp': _byte_features(line_simp),
            'ppdb_score': _float_features([ppdb_score]),
            'dsim_score': _float_features([dsim_score]),
            'add_score':  _float_features([add_score]),
            'len_score': _float_features([len_score]),
            'ppdb_rule': _byte_features(line_ppdbrule)
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        cnt += 1
    print('Finished shard id %s with size %s' % (shard_id, cnt))
    time_span = datetime.now() - s_time
    print('Done id:%s with time span %s' % (shard_id, time_span))
    writer.close()


if __name__ == '__main__':
    p = Pool(10)
    p.map(generate_example, range(1280))