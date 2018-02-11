from nltk.translate.bleu_score import sentence_bleu

ref = [['c1', 'c2', 's3', 'c4', 'c5', 'c6', 'x'],['c1', 's2', 'c3', 's4', 'c5', 'c6', 'x'], ['s1', 'c2', 'c3', 'c4', 'c5', 'c6']]
out2 = ['c1','c2','c3', 'c4', 'c5', 'c6']
bleu1 = sentence_bleu(ref, out2, weights=[1])
print(bleu1)


out2 = ['s1','s2','s3', 'c4', 'c5']
bleu2 = sentence_bleu(ref, out2, weights=[1])
print(bleu2)