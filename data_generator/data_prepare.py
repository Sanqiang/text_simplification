"""DEPRECATED: I plan to move the data into memory that ease the shuffle as well as access.
    Convert the raw text file into TFRecords"""
from util import constant
import tensorflow as tf

class DataPrepareBase:
    def __init__(self, data_simple, data_complex, output):
        self.data_simple = data_simple
        self.data_complex = data_complex
        self.output = output

    def Count(self, add_assert=False):
        count = 0
        for _ in open(self.data_simple):
            count += 1
        if add_assert:
            for _ in open(self.data_complex):
                count -= 1
            assert count == 0
        return count

    def _ByteFeature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def Convert(self, num_examples=-1):
        f_simple = open(self.data_simple)
        f_complex = open(self.data_complex)
        if num_examples == -1:
            num_examples = self.Count()

        writer = tf.python_io.TFRecordWriter(self.output)
        for index in range(num_examples):
            line_simple = next(f_simple, None)
            line_complex = next(f_complex, None)
            if line_simple is None:
                assert line_complex is None and num_examples == 0
                break

            example = tf.train.Example(features=tf.train.Features(feature={
                constant.SIMPLE_SENTENCE_LABEL:
                    self._ByteFeature(line_simple.strip().encode()),
                constant.COMPLEX_SENTENCE_LABEL:
                    self._ByteFeature(line_complex.strip().encode())}))
            writer.write(example.SerializeToString())
        writer.close()

if __name__ == '__main__':
    data = DataPrepareBase('../data/dummy_simple_dataset', '../data/dummy_complex_dataset',
                           '../data/dummy_dataset')
    data.Convert()



