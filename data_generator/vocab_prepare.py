"""Prepare the vocabulary list file for model training and validation.
   Note the file is independet with model."""
from collections import Counter

from nltk import word_tokenize

from data_generator.vocab import Vocab
from model.model_config import WikiDressLargeDefault, DefaultConfig, list_config
from util.arguments import get_args
from util.data.text_encoder import SubwordTextEncoder
from util.data import text_encoder

args = get_args()


class VocabPrepare:
    def __init__(self, data_file, output, model_config, data_file2=None):
        """data_file: original data file
           output: outputed vocab file
           data_file2: additional original data file for tied embedding
        """
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

        if self.model_config.subword_vocab_size <= 0:
            c = c.most_common(len(c))

            writer = open(self.output, 'w', encoding='utf-8')
            for word, cnt in c:
                writer.write(word)
                writer.write('\t')
                writer.write(str(cnt))
                writer.write('\n')
            writer.close()
            print('Processed vocab with size %d' % len(c))
        else:
            sub_word = SubwordTextEncoder.build_to_target_size(self.model_config.subword_vocab_size, c, 1, 1e3,
                                                               num_iterations=100)
            for i, subtoken_string in enumerate(sub_word._all_subtoken_strings):
                if subtoken_string in text_encoder.RESERVED_TOKENS_DICT:
                    sub_word._all_subtoken_strings[i] = subtoken_string + "_"
            sub_word.store_to_file(self.output)
            print('Processed vocab with size %d' % len(sub_word._all_subtoken_strings))


if __name__ == '__main__':
    config = None
    if args.mode == 'dummy':
        config = DefaultConfig()
    elif args.mode == 'dress':
        config = WikiDressLargeDefault()
    print(list_config(config))

    if config.subword_vocab_size <= 0:
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
    else:
        voc = VocabPrepare(config.train_dataset_complex,
                           config.subword_vocab_complex,
                           config)
        voc.prepare_vocab()
        voc = VocabPrepare(config.train_dataset_simple,
                           config.subword_vocab_simple,
                           config)
        voc.prepare_vocab()

        voc = VocabPrepare(config.train_dataset_complex,
                           config.subword_vocab_all,
                           config,
                           config.train_dataset_simple)
        voc.prepare_vocab()

