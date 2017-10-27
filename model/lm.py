import tensorflow as tf
import numpy as np
from google.protobuf import text_format

from model.model_config import get_path


MAX_WORD_LEN = 50
BASE_PATH = get_path('../text_simplification_data/lm1b/')


class GoogleLM:
    """Get from https://github.com/tensorflow/models/tree/master/research/lm_1b."""
    def __init__(self):
        self.vocab = CharsVocabulary(BASE_PATH + 'vocab-2016-09-10.txt', MAX_WORD_LEN)
        self.sess, self.t = self.load_model()
        print('Init GoogleLM Session .')

    def get_weight(self, sentence):
        inputs, targets, weights, char_inputs = self.get_data(sentence)
        output_weights = []
        for inp, target, weight, char_input in zip(
                inputs[:-2], targets[:-2], weights[:-2], char_inputs[:-2]):
            input_dict = {
                self.t['inputs_in']: inp,
                self.t['targets_in']: target,
                self.t['target_weights_in']: weight}
            if 'char_inputs_in' in self.t:
                input_dict[self.t['char_inputs_in']] = char_input
            log_perp = self.sess.run(self.t['log_perplexity_out'], feed_dict=input_dict)
            output_weight = 1 + 1 / log_perp
            if output_weight > 5:
                output_weight = 5
            output_weights.append(output_weight)
        return np.mean(output_weights)

    def assess(self, sentence):
        inputs, targets, weights, char_inputs = self.get_data(sentence)
        step = 1
        for inp, target, weight, char_input in zip(
                inputs[:-1], targets[:-1], weights[:-1], char_inputs[:-1]):
            input_dict = {
                self.t['inputs_in']: inp,
                self.t['targets_in']: target,
                self.t['target_weights_in']: weight}
            if 'char_inputs_in' in self.t:
                input_dict[self.t['char_inputs_in']] = char_input
            log_perp = self.sess.run(self.t['log_perplexity_out'], feed_dict=input_dict)

            perplexity = 1 + 1 / log_perp

            print('Assess step %d, Barrier Value for word %s is %s' %
                  (step, self.vocab.id_to_word(target[0][0]), perplexity))
            step += 1

    def assess_interactive(self):
        while True:
            sentence = input('Type in the document you want to assess?\n')
            self.assess(sentence)

    def get_data(self, sentence):
        num_words = 2 + len(sentence.split())
        chars_ids = [np.expand_dims(x, axis=0) for x in np.split(self.vocab.encode_chars(sentence), num_words, 0)]
        ids = np.split(np.expand_dims(self.vocab.encode(sentence), axis=0), num_words, -1)
        inputs = ids
        targets = ids[1:] + [np.zeros((1, 1))]
        weights = [np.ones((1, 1))] * num_words
        return inputs, targets, weights, chars_ids

    def load_model(self):
        """Load the model from GraphDef and Checkpoint.

        Args:
          gd_file: GraphDef proto text file.
          ckpt_file: TensorFlow Checkpoint file.

        Returns:
          TensorFlow session and tensors dict.
        """
        lm_graph = tf.Graph()
        with lm_graph.as_default():
            with tf.gfile.FastGFile(BASE_PATH + 'graph-2016-09-10.pbtxt', 'r') as f:
                s = f.read()
                gd = tf.GraphDef()
                text_format.Merge(s, gd)

            t = {}
            [t['states_init'], t['lstm/lstm_0/control_dependency'],
             t['lstm/lstm_1/control_dependency'], t['softmax_out'], t['class_ids_out'],
             t['class_weights_out'], t['log_perplexity_out'], t['inputs_in'],
             t['targets_in'], t['target_weights_in'], t['char_inputs_in'],
             t['all_embs'], t['softmax_weights'], t['global_step']
             ] = tf.import_graph_def(gd, {}, ['states_init',
                                              'lstm/lstm_0/control_dependency:0',
                                              'lstm/lstm_1/control_dependency:0',
                                              'softmax_out:0',
                                              'class_ids_out:0',
                                              'class_weights_out:0',
                                              'log_perplexity_out:0',
                                              'inputs_in:0',
                                              'targets_in:0',
                                              'target_weights_in:0',
                                              'char_inputs_in:0',
                                              'all_embs_out:0',
                                              'Reshape_3:0',
                                              'global_step:0'], name='')

            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=lm_graph)
            sess.run('save/restore_all', {'save/Const:0': BASE_PATH + 'ckpt-*'})
            sess.run(t['states_init'])

        return sess, t




class Vocabulary(object):
    """Class that holds a vocabulary for the dataset."""

    def __init__(self, filename):
        """Initialize vocabulary.

        Args:
          filename: Vocabulary file name.
        """

        self._id_to_word = []
        self._word_to_id = {}
        self._unk = -1
        self._bos = -1
        self._eos = -1

        with tf.gfile.Open(filename) as f:
            idx = 0
            for line in f:
                word_name = line.strip()
                if word_name == '<S>':
                    self._bos = idx
                elif word_name == '</S>':
                    self._eos = idx
                elif word_name == '<UNK>':
                    self._unk = idx
                if word_name == '!!!MAXTERMID':
                    continue

                self._id_to_word.append(word_name)
                self._word_to_id[word_name] = idx
                idx += 1

    @property
    def bos(self):
        return self._bos

    @property
    def eos(self):
        return self._eos

    @property
    def unk(self):
        return self._unk

    @property
    def size(self):
        return len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self.unk

    def id_to_word(self, cur_id):
        if cur_id < self.size:
            return self._id_to_word[cur_id]
        return 'ERROR'

    def decode(self, cur_ids):
        """Convert a list of ids to a sentence, with space inserted."""
        return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

    def encode(self, sentence):
        """Convert a sentence to a list of ids, with special tokens added."""
        word_ids = [self.word_to_id(cur_word) for cur_word in sentence.split()]
        return np.array([self.bos] + word_ids + [self.eos], dtype=np.int32)


class CharsVocabulary(Vocabulary):
    """Vocabulary containing character-level information."""

    def __init__(self, filename, max_word_length):
        super(CharsVocabulary, self).__init__(filename)
        self._max_word_length = max_word_length
        chars_set = set()

        for word in self._id_to_word:
            chars_set |= set(word)

        free_ids = []
        for i in range(256):
            if chr(i) in chars_set:
                continue
            free_ids.append(chr(i))

        if len(free_ids) < 5:
            raise ValueError('Not enough free char ids: %d' % len(free_ids))

        self.bos_char = free_ids[0]  # <begin sentence>
        self.eos_char = free_ids[1]  # <end sentence>
        self.bow_char = free_ids[2]  # <begin word>
        self.eow_char = free_ids[3]  # <end word>
        self.pad_char = free_ids[4]  # <padding>

        chars_set |= {self.bos_char, self.eos_char, self.bow_char, self.eow_char,
                      self.pad_char}

        self._char_set = chars_set
        num_words = len(self._id_to_word)

        self._word_char_ids = np.zeros([num_words, max_word_length], dtype=np.int32)

        self.bos_chars = self._convert_word_to_char_ids(self.bos_char)
        self.eos_chars = self._convert_word_to_char_ids(self.eos_char)

        for i, word in enumerate(self._id_to_word):
            self._word_char_ids[i] = self._convert_word_to_char_ids(word)

    @property
    def word_char_ids(self):
        return self._word_char_ids

    @property
    def max_word_length(self):
        return self._max_word_length

    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self.max_word_length], dtype=np.int32)
        code[:] = ord(self.pad_char)

        if len(word) > self.max_word_length - 2:
            word = word[:self.max_word_length - 2]
        cur_word = self.bow_char + word + self.eow_char
        for j in range(len(cur_word)):
            code[j] = ord(cur_word[j])
        return code

    def word_to_char_ids(self, word):
        if word in self._word_to_id:
            return self._word_char_ids[self._word_to_id[word]]
        else:
            return self._convert_word_to_char_ids(word)

    def encode_chars(self, sentence):
        chars_ids = [self.word_to_char_ids(cur_word)
                     for cur_word in sentence.split()]
        return np.vstack([self.bos_chars] + chars_ids + [self.eos_chars])


if __name__ == '__main__':
    lm = GoogleLM()
    # lm.assess_interactive()
    lm.get_weight('i am a sanqiang .')