"""Get max lens for subvoc dataset for trans data"""
from data_generator.vocab import Vocab
from model.model_config import WikiTransTrainConfig

vocab_comp = Vocab(
    WikiTransTrainConfig(), '/Users/sanqiangzhao/git/text_simplification_data/vocab/comp30k.subvocab')
vocab_simp = Vocab(
    WikiTransTrainConfig(), '/Users/sanqiangzhao/git/text_simplification_data/vocab/simp30k.subvocab')

max_l_comp, max_l_simp = 0, 0
for line in open('/Users/sanqiangzhao/git/text_simplification_data/val_0930/words_comps'):
    l_comp = len(vocab_comp.encode(line))
    l_simp = len(vocab_simp.encode(line))
    max_l_comp = max(max_l_comp, l_comp)
    max_l_simp = max(max_l_simp, l_simp)

print(max_l_comp)
print(max_l_simp)
