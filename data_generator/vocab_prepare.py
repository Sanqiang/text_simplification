"""Prepare the vocabulary list file for model training and validation.
   Note the file is independet with model."""
from collections import Counter

from data_generator.vocab import Vocab
from model.model_config import WikiDressLargeDefault

from nltk import word_tokenize

class VocabPrepare:
    def __init__(self, data_file, output, model_config):
        self.data_file = data_file
        self.output = output
        self.model_config = model_config

    def prepare_vocab(self):
        c = Counter()
        for line in open(self.data_file, encoding='utf-8'):
            if self.model_config.tokenizer == 'split':
                words = line.split()
            elif self.model_config.tokenizer == 'nltk':
                words = word_tokenize(line)
            else:
                raise Exception('Unknown tokenizer.')
            words = [Vocab.process_word(word, self.model_config)
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
        print('Processed vocab with size %d' % len(c))


if __name__ == '__main__':
    voc = VocabPrepare('../../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src',
                       '../../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src.vocab',
                       WikiDressLargeDefault())
    voc.prepare_vocab()
    voc = VocabPrepare('../../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.dst',
                       '../../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.dst.vocab',
                       WikiDressLargeDefault())
    voc.prepare_vocab()
