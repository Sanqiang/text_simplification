from data_generator.vocab import Vocab
from util import constant
from types import SimpleNamespace
from collections import Counter

model_config = SimpleNamespace(min_count=0, subword_vocab_size=50000, lower_case=True)
vocab = Vocab(model_config, '/Users/zhaosanqiang916/git/text_simplification_data/wiki/voc/voc_comp_sub30k.txt')
f = open('/Users/zhaosanqiang916/git/text_simplification_data/test/wiki.full.aner.test.src')
c = Counter()
for line in f:
    words = line.lower().split()
    words = [constant.SYMBOL_START] + words + [constant.SYMBOL_END]
    words = vocab.encode(' '.join(words))
    c.update([len(words)])
    print(len(words))

print(c)