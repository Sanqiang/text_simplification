"""
Deprecated: Will train our new language model in language_model folder.
"""
import tensorflow as tf
import numpy as np
from collections import defaultdict
from google.protobuf import text_format
import time

from model.model_config import get_path


MAX_WORD_LEN = 50
BASE_PATH = get_path('../text_simplification_data/lm1b/')


class GoogleLM:
    """Get from https://github.com/tensorflow/models/tree/master/research/lm_1b."""
    def __init__(self, batch_size=32):
        self.vocab = CharsVocabulary(BASE_PATH + 'vocab-2016-09-10.txt', MAX_WORD_LEN)
        self.sess, self.t = self.load_model()
        print('Init GoogleLM Session .')

    def get_batch_weight(self, sentneces, num_steps):
        inputs, targets, weights, char_inputs = self.get_batch_data(sentneces, num_steps)
        log_perps = []
        for inp, target, weight, char_input in zip(
                inputs, targets, weights, char_inputs):
            input_dict = {
                self.t['inputs_in']: inp,
                self.t['targets_in']: target,
                self.t['target_weights_in']: weight}
            if 'char_inputs_in' in self.t:
                input_dict[self.t['char_inputs_in']] = char_input
            log_perp = self.sess.run(self.t['log_perplexity_out'], feed_dict=input_dict)
            log_perps.append(log_perp)
        return np.mean(log_perps)

    def get_batch_data(self, sentences, num_steps, eos_id=0, bos_id=1):
        chars_ids = [self.vocab.encode_chars(sentence) for sentence in sentences]
        input_sent_ids = [self.vocab.encode(sentence) for sentence in sentences]
        target_sent_ids = []
        weights = []
        for i, sent_id in enumerate(input_sent_ids):
            weights.append([1.0] * len(sent_id))
            while len(sent_id) < num_steps:
                input_sent_ids[i].append(eos_id)
                weights[i].append(0.0)
                chars_ids[i].append(np.zeros(50,))
            target_sent_ids.append(np.concatenate([sent_id[1:], [eos_id]]))

        input_sent_ids = np.split(np.stack(input_sent_ids), num_steps, -1)
        target_sent_ids = np.split(np.stack(target_sent_ids), num_steps, -1)
        weights = np.split(np.stack(weights), num_steps, -1)
        chars_ids = np.split(np.stack(chars_ids), num_steps, -1)
        return input_sent_ids, target_sent_ids, weights, chars_ids



    def get_weight(self, sentence):
        inputs, targets, weights, char_inputs = self.get_data(sentence)
        log_perps = []
        for inp, target, weight, char_input in zip(
                inputs[:-2], targets[:-2], weights[:-2], char_inputs[:-2]):
            input_dict = {
                self.t['inputs_in']: inp,
                self.t['targets_in']: target,
                self.t['target_weights_in']: weight}
            if 'char_inputs_in' in self.t:
                input_dict[self.t['char_inputs_in']] = char_input
            log_perp = self.sess.run(self.t['log_perplexity_out'], feed_dict=input_dict)
            log_perps.append(log_perp)
        return np.mean(log_perps)

    def assess(self, sentence):
        inputs, targets, weights, char_inputs = self.get_data(sentence)
        step = 1
        perplexitys = []
        for inp, target, weight, char_input in zip(
                inputs[:-1], targets[:-1], weights[:-1], char_inputs[:-1]):
            input_dict = {
                self.t['inputs_in']: inp,
                self.t['targets_in']: target,
                self.t['target_weights_in']: weight}
            if 'char_inputs_in' in self.t:
                input_dict[self.t['char_inputs_in']] = char_input
            log_perp = self.sess.run(self.t['log_perplexity_out'], feed_dict=input_dict)

            perplexity = log_perp
            perplexitys.append(perplexity)

            print('Assess step %d, Barrier Value for word %s is %s' %
                  (step, self.vocab.id_to_word(target[0][0]), perplexity))
            step += 1
        print('Final Loss\t%s.' % np.mean(perplexitys))

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


class NgramLM:
    def __init__(self):
        self.generate_data_file()

    def generate_data_file(self, file='googlebooks-eng-all-3gram-20120701-en'):
        grams = defaultdict(list)
        gram_path = get_path(
            '../text_simplification_data/wiki/' + file)
        idx = 0
        pre_time = time.time()
        for line in open(gram_path, encoding='utf-8'):
            items = line.split('\t')
            ngram = '|'.join([word.split('_')[0] for word in items[0].split(' ')])
            # year = int(items[1])
            cnt = int(items[2])
            grams[ngram].append(cnt)
            idx += 1
            if idx % 1000000 == 0:
                cur_time = time.time()
                print('processed %s. in %s' % (idx, cur_time - pre_time))
                pre_time = cur_time

        output_file = get_path(
            '../text_simplification_data/wiki/' + file + '.processed')
        f = open(output_file, 'w', encoding='utf-8')
        outputs = []
        for ngram in grams:
            output = '\t'.join([ngram, str(np.mean(grams[ngram]))])
            outputs.append(output)
            if len(outputs) == 100000:
                f.write('\n'.join(outputs))
                f.flush()
                outputs = []
        f.write('\n'.join(outputs))
        f.flush()
        f.close()

if __name__ == '__main__':
    # lm = NgramLM()

    lm = GoogleLM()
    lm.get_batch_weight(['car drives .', 'car flies .', 'car dies .'], 5)
    # lm.assess_interactive()
    # lm.get_weight('i am a sanqiang .')