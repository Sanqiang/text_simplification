"""
For wikisplit, provide tf.example for training
"""
import tensorflow as tf


def _byte_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(value, 'utf-8')]))


writer = tf.python_io.TFRecordWriter(
    '/Users/sanqiang/git/ts/text_simplification_data/newsela/train/train.tfexample')

file_src = '/Users/sanqiang/git/ts/text_simplification_data/newsela/train/train.src'
file_dst = '/Users/sanqiang/git/ts/text_simplification_data/newsela/train/train.dst'
lines_src, lines_dst = open(file_src).readlines(), open(file_dst).readlines()
assert len(lines_dst) == len(lines_src)
for line_src, line_dst in zip(lines_src, lines_dst):
    feature = {
        'line_comp': _byte_features(line_src),
        'line_simp': _byte_features(line_dst),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
writer.close()