"""Check FKGL for ground truth."""
from util.fkgl import Readability


# path = '/Users/zhaosanqiang916/git/text_simplification_data/test/test.8turkers.tok.norm'
path = '/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.dst'
text = open(path, encoding='utf-8').read()
rd = Readability(text)
fkgl = rd.FleschKincaidGradeLevel()
print(fkgl)

# Report:
# For 8 ref dataset: 9.5088 -> 8.0282
# For Dress train: 11.3091 -> 8.2863