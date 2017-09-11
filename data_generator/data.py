import random as rd

from data_generator.vocab import Vocab
from nltk import word_tokenize

class Data:
    def __init__(self, data_simple_path, data_complex_path, vocab_simple_path, vocab_complex_path):
        self.vocab_simple = Vocab(vocab_simple_path)
        self.vocab_complex = Vocab(vocab_complex_path)
        self.data_simple = self.PopulateData(data_simple_path, self.vocab_simple)
        self.data_complex = self.PopulateData(data_complex_path, self.vocab_complex)
        self.size = len(self.data_simple)

    def PopulateData(self, data_path, vocab):
        # Populate data into memory
        data = []
        for line in open(data_path):
            words = word_tokenize(line)
            words = [Vocab.ProcessWord(word)
                     for word in words]
            words = [vocab.Encode(word) for word in words]
            data.append(words)
        return data

    def GetDataSample(self):
        i = rd.sample(range(self.size), 1)[0]
        return self.data_simple[i], self.data_complex[i]

