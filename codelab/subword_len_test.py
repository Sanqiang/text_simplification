from data_generator.vocab import Vocab
from util import constant
from types import SimpleNamespace
from collections import Counter

model_config = SimpleNamespace(min_count=0, subword_vocab_size=0, lower_case=True, bert_mode=['bert_token'])
vocab = Vocab(model_config, '/zfs1/hdaqing/saz31/dataset/vocab/bert/vocab_30k')
# print(vocab.encode('#sep# -lrb- . #pad#'))
# vocab = Vocab(model_config, '/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src.vocab.lower')
f = open('/Users/sanqiang/git/ts/text_simplification_data/val2/ncomp/tune.8turkers.tok.norm.ori')
c = Counter()
cnt = 0
tcnt = 0
max_len = 0
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
    max_len = max(max_len, len(words))
    c.update([len(words)])

print(c)
print(max_len)