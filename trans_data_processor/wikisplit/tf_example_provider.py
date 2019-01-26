"""
For wikisplit, provide tf.example for training
"""
import tensorflow as tf


def _byte_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(value, 'utf-8')]))


writer = tf.python_io.TFRecordWriter(
    '/Users/sanqiang/git/ts/text_simplification_data/wikisplit/train_tfexample/train.tfexample')

tag = 'train'
tsv_file_src = '/Users/sanqiang/git/ts/text_simplification_data/wikisplit/train/src_%s.tsv' % tag
tsv_file_dst = '/Users/sanqiang/git/ts/text_simplification_data/wikisplit/train/dst_%s.tsv' % tag
lines_src, lines_dst = open(tsv_file_src).readlines(), open(tsv_file_dst).readlines()
assert len(lines_dst) == len(lines_src)
for line_src, line_dst in zip(lines_src, lines_dst):
    feature = {
        'line_comp': _byte_features(line_src),
        'line_simp': _byte_features(line_dst),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
writer.close()