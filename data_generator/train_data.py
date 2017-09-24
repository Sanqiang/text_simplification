import random as rd
import copy as cp
import numpy as np

from data_generator.vocab import Vocab
from nltk import word_tokenize
from util import constant


class TrainData:
    def __init__(self, model_config):
        self.model_config = model_config
        vocab_simple_path = self.model_config.vocab_simple
        vocab_complex_path = self.model_config.vocab_complex
        vocab_all_path = self.model_config.vocab_all
        data_simple_path = self.model_config.train_dataset_simple
        data_complex_path = self.model_config.train_dataset_complex

        if (self.model_config.tie_embedding == 'none' or
                    self.model_config.tie_embedding == 'dec_out'):
            self.vocab_simple = Vocab(model_config, vocab_simple_path)
            self.vocab_complex = Vocab(model_config, vocab_complex_path)
        elif (self.model_config.tie_embedding == 'all' or
                    self.model_config.tie_embedding == 'enc_dec'):
            self.vocab_simple = Vocab(model_config, vocab_all_path)
            self.vocab_complex = Vocab(model_config, vocab_all_path)

        # Populate basic complex simple pairs
        self.data_simple = self.populate_data(data_simple_path, self.vocab_simple)
        self.data_complex = self.populate_data(data_complex_path, self.vocab_complex)
        self.size = len(self.data_simple)
        print('Use Train Dataset: \n Simple\t %s. \n Complex\t %s. \n Size\t %d'
              % (data_simple_path, data_complex_path, self.size))
        self.init_pretrained_embedding()

    def populate_data(self, data_path, vocab):
        # Populate data into memory
        data = []
        for line in open(data_path, encoding='utf-8'):
            # line = line.split('\t')[2]
            if self.model_config.tokenizer == 'split':
                words = line.split()
            elif self.model_config.tokenizer == 'nltk':
                words = word_tokenize(line)
            else:
                raise Exception('Unknown tokenizer.')

            words = [Vocab.process_word(word, self.model_config)
                     for word in words]
            words = [vocab.encode(word) for word in words]
            words = ([self.vocab_simple.encode(constant.SYMBOL_START)] + words +
                     [self.vocab_simple.encode(constant.SYMBOL_END)])

            data.append(words)
        return data

    def get_data_sample(self):
        i = rd.sample(range(self.size), 1)[0]
        return cp.deepcopy(self.data_simple[i]), cp.deepcopy(self.data_complex[i])

    def get_data_iter(self):
        i = 0
        while True:
            yield cp.deepcopy(self.data_simple[i]), cp.deepcopy(self.data_complex[i])
            i += 1
            if i == len(self.data_simple):
                yield None, None

    def init_pretrained_embedding(self):
        if self.model_config.pretrained_embedding is None:
            return

        print('Use Pretrained Embedding\t%s.' % self.model_config.pretrained_embedding)

        # if (os.path.exists(self.model_config.pretrained_embedding_complex + '.npy') and
        #         os.path.exists(self.model_config.pretrained_embedding_simple + '.npy')):
        #     self.pretrained_emb_complex = np.load(self.model_config.pretrained_embedding_complex + '.npy')
        #     self.pretrained_emb_simple = np.load(self.model_config.pretrained_embedding_simple + '.npy')


        if not hasattr(self, 'glove'):
            self.glove = {}
            for line in open(self.model_config.pretrained_embedding, encoding='utf-8'):
                pairs = line.split()
                word = ' '.join(pairs[:-self.model_config.dimension])
                if word in self.vocab_simple.w2i or word in self.vocab_complex.w2i:
                    embedding = pairs[-self.model_config.dimension:]
                    self.glove[word] = embedding

            # For vocabulary complex
            pretrained_cnt = 0
            random_cnt = 0
            self.pretrained_emb_complex = np.empty(
                (len(self.vocab_complex.i2w), self.model_config.dimension), dtype=np.float32)
            for wid, word in enumerate(self.vocab_complex.i2w):
                if word in self.glove:
                    n_vector = np.array(self.glove[word])

                    self.pretrained_emb_complex[wid, :] = n_vector
                    pretrained_cnt += 1
                else:
                    n_vector = np.array([np.random.uniform(-0.08, 0.08)
                                         for _ in range(self.model_config.dimension)])
                    self.pretrained_emb_complex[wid, :] = n_vector
                    random_cnt += 1
            assert len(self.vocab_complex.i2w) ==  random_cnt + pretrained_cnt
            print(
                'For Vocab Complex, %s words initialized with pretrained vector, '
                'other %s words initialized randomly. Save to %s.' %
                (pretrained_cnt, random_cnt, self.model_config.pretrained_embedding_complex))
            np.save(self.model_config.pretrained_embedding_complex, self.pretrained_emb_complex)

            # For vocabulary simple
            pretrained_cnt = 0
            random_cnt = 0
            self.pretrained_emb_simple = np.empty(
                (len(self.vocab_simple.i2w), self.model_config.dimension), dtype=np.float32)
            for wid, word in enumerate(self.vocab_simple.i2w):
                if word in self.glove:
                    n_vector = np.array(self.glove[word])
                    self.pretrained_emb_simple[wid, :] = n_vector
                    pretrained_cnt += 1
                else:
                    n_vector = np.array([np.random.uniform(-0.08, 0.08)
                                         for _ in range(self.model_config.dimension)])
                    self.pretrained_emb_simple[wid, :] = n_vector
                    random_cnt += 1
            assert len(self.vocab_simple.i2w) == random_cnt + pretrained_cnt
            print(
                'For Vocab Simple, %s words initialized with pretrained vector, '
                'other %s words initialized randomly. Save to %s.' %
                (pretrained_cnt, random_cnt, self.model_config.pretrained_embedding_simple))
            np.save(self.model_config.pretrained_embedding_simple, self.pretrained_emb_simple)