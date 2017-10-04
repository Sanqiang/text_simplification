"""Prepare the vocabulary list file for model training and validation.
   Note the file is independet with model."""
from collections import Counter

from data_generator.vocab import Vocab
from model.model_config import WikiDressLargeDefault, DefaultConfig, list_config
from util.arguments import get_args
from nltk import word_tokenize



args = get_args()


class VocabPrepare:
    def __init__(self, data_file, output, model_config, data_file2=None):
        self.data_file = data_file
        self.data_file2 = data_file2
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

        if self.data_file2 is not None:
            for line in open(self.data_file2, encoding='utf-8'):
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

        writer = open(self.output, 'w', encoding='utf-8')
        for word, cnt in c:
            writer.write(word)
            writer.write('\t')
            writer.write(str(cnt))
            writer.write('\n')
        writer.close()
        print('Processed vocab with size %d' % len(c))



if __name__ == '__main__':
    config = None
    if args.mode == 'dummy':
        config = DefaultConfig()
    elif args.mode == 'dress':
        config = WikiDressLargeDefault()
    print(list_config(config))

    voc = VocabPrepare(config.train_dataset_complex,
                       config.vocab_complex,
                       config)
    voc.prepare_vocab()
    voc = VocabPrepare(config.train_dataset_simple,
                       config.vocab_simple,
                       config)
    voc.prepare_vocab()

    voc = VocabPrepare(config.train_dataset_complex,
                       config.vocab_all,
                       config,
                       config.train_dataset_simple)
    voc.prepare_vocab()

