from collections import Counter

from data_generator.vocab import Vocab
from data_generator.vocab_config import DefaultConfig

from nltk import word_tokenize

class VocabPrepare:
    def __init__(self, data_file, output, voc_config=None):
        self.data_file = data_file
        self.output = output
        self.voc_config = (DefaultConfig()
                           if voc_config is None else voc_config)

    def prepare_vocab(self):
        c = Counter()
        for line in open(self.data_file):
            words = word_tokenize(line)
            words = [Vocab.process_word(word)
                     for word in words]
            c.update(words)

        c = c.most_common(len(c))

        writer = open(self.output, 'w')
        for word, cnt in c:
            writer.write(word)
            writer.write('\t')
            writer.write(str(cnt))
            writer.write('\n')
        writer.close()


if __name__ == '__main__':
    # voc = VocabPrepare('../data/dummy_complex_dataset', '../data/dummy_complex_vocab')
    # voc.PrepareVocab()
    voc = VocabPrepare('../data/dummy_simple_dataset', '../data/dummy_simple_vocab')
    voc.prepare_vocab()
