from nltk.translate.bleu_score import sentence_bleu

x = 'a ccc ddd rrr a a a ss'
y = 'a ccc ddd eee a a a ee'
print(sentence_bleu([x], y))
print(sentence_bleu([y], x))