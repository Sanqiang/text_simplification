from nltk.translate.bleu_score import sentence_bleu
import numpy as np

comp_lines = open(
    '/Users/zhaosanqiang916/git/text_simplification_data/test/wiki.full.aner.ori.test.src').readlines()
output_path = '/zfs1/hdaqing/saz31/text_simplification/dress_final_ffn_cl0/result/eightref_test_ppdbe_v2/joshua_target_1159549.txt'
lines = open(output_path).readlines()

assert len(lines) == len(comp_lines)

bleus = []
for lid in range(len(lines)):
    line = lines[lid].strip().lower().split()
    comp_line = comp_lines[lid].strip().lower().split()

    try:
        bleu = sentence_bleu([comp_line], line)
    except:
        bleu = 0
    bleus.append(bleu)

print('overlap = ' + str(np.mean(bleus)))