from data_generator.vocab import Vocab
from util import constant
from types import SimpleNamespace
from collections import Counter

model_config = SimpleNamespace(min_count=0, subword_vocab_size=50000, lower_case=True)
vocab = Vocab(model_config, '/Users/zhaosanqiang916/git/text_simplification_data/wiki/voc/voc_all_sub50k.txt')
# vocab = Vocab(model_config, '/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src.vocab.lower')
f = open('/Users/zhaosanqiang916/git/text_simplification_data/wiki/ner3/ner_comp.txt')
c = Counter()
cnt = 0
tcnt = 0
for line in f:
    words = line.lower().split()
    words = [constant.SYMBOL_START] + words + [constant.SYMBOL_END]
    words = vocab.encode(' '.join(words))
    tcnt += 1
    if len(words) > 200:
        # c.update([len(words)])
        # print(len(words))
        cnt += 1
    if tcnt % 100000 == 0:
        print(cnt/tcnt)

print(c)
print(cnt)