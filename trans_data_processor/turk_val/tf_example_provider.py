"""
For turk val (post train), provide tf.example for training
"""
import tensorflow as tf


def _byte_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(value, 'utf-8')]))


writer = tf.python_io.TFRecordWriter(
    '/Users/sanqiang/git/ts/text_simplification_data/val2/train_tfexample/train.tfexample')

tsv_file_src = '/Users/sanqiang/git/ts/text_simplification_data/val2/ncomp/tune.8turkers.tok.norm.ori'
tsv_file_dst = '/Users/sanqiang/git/ts/text_simplification_data/val2/nsimp/tune.8turkers.tok.turk.'
for tid in range(8):
    lines_src, lines_dst = open(tsv_file_src).readlines(), open(tsv_file_dst + str(tid)).readlines()
    assert len(lines_dst) == len(lines_src)
    for line_src, line_dst in zip(lines_src, lines_dst):
        feature = {
            'line_comp': _byte_features(line_src),
            'line_simp': _byte_features(line_dst),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
writer.close()