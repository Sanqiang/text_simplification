from data_generator.vocab import Vocab
from util import constant
from types import SimpleNamespace
from collections import Counter

model_config = SimpleNamespace(min_count=0, subword_vocab_size=50000, lower_case=True)
vocab = Vocab(model_config, '/zfs1/hdaqing/saz31/dataset/vocab/all30k.subvocab')
ids = vocab.encode(constant.SYMBOL_START + ' -lrb- . #pad# #pad# #pad# #pad# #pad#')

print(ids)
print('=====')


print([vocab.describe([id]) for id in ids])
print(vocab.describe(ids))

print([vocab.subword.all_subtoken_strings[id] for id in ids])


