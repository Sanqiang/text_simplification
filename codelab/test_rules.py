from model.model_config import SubTestWikiEightRefConfigV2Sing
import numpy as np
from collections import defaultdict

model_config = SubTestWikiEightRefConfigV2Sing()
output_path = '/zfs1/hdaqing/saz31/text_simplification/dress_final_ffn_cl2/result/eightref_test_v2singdfs/joshua_target_389906.txt'

ppdb_mapper = defaultdict(set)
ppdb_path = model_config.path_ppdb_refine #'/Users/zhaosanqiang916/git/text_simplification_data/ppdb/XU_PPDB'
for line in open(ppdb_path):
    line = line.strip().lower()
    items = line.split('\t')
    if len(items) < 5:
        continue
    ppdb_mapper[items[3]].add(items[4])

# Get correct rules to be applied by filter out based on gt reference
rule_lines = open(model_config.val_dataset_complex_ppdb).readlines()
complex_lines = open(model_config.val_dataset_complex).readlines()

ref_lines_col = []
for ref_i in range(model_config.num_refs):
    tmp_path = model_config.val_dataset_simple_folder + model_config.val_dataset_simple_rawlines_file_references + str(ref_i)
    tmp_ref_lines = open(tmp_path).readlines()
    ref_lines_col.append([l.strip().lower().split() for l in tmp_ref_lines])

target_words_col, non_target_words_col = [], []
for lid, rule_line in enumerate(rule_lines):
    rule_line = rule_line.strip().lower()
    cur_target_words, cur_non_target_words_col = [], []
    rules = rule_line.split('\t')
    for rule in rules:
        items = rule.split('=>')
        if len(items) == 4:
            target_words = items[2]
            valid = any([target_words in ref_lines_col[ref_i][lid] for ref_i in range(model_config.num_refs)])
            if valid:
                cur_target_words.append(target_words)
        complex_line = complex_lines[lid].strip().lower().split()
        for origin_words in complex_line:
            for non_target_words in ppdb_mapper[origin_words] - set(cur_target_words):
                cur_non_target_words_col.append(non_target_words)
    for ref_i in range(model_config.num_refs):
        cur_non_target_words_col = list(set(cur_non_target_words_col) - set(ref_lines_col[ref_i][lid]))
    cur_non_target_words_col = list(set(cur_non_target_words_col) - set(cur_target_words))
    target_words_col.append(cur_target_words)
    non_target_words_col.append(cur_non_target_words_col)

# Evaluation
precs = []
recalls = []
f1s = []
out_lines = open(output_path).readlines()
for lid, line in enumerate(out_lines):
    if lid == 295:
        x=1
    line = line.strip().lower()
    cnt_tp, cnt_fp = 1, 1
    cnt_fn, cnt_tn = 1, 1
    target_words = target_words_col[lid]
    non_target_words = non_target_words_col[lid]
    words = line.split()
    for target_word in target_words:
        if target_word in words:
            cnt_tp += 1 # correctly use ppdb rules
        else:
            cnt_fn += 1 # miss the corrected ppdb rules
    for target_word in non_target_words:
        if target_word in words:
            cnt_fp += 1 # incorrectly use ppdb rules
        else:
            cnt_tn += 1 # miss the incorrected ppdb rules
    prec = cnt_tp / (cnt_tp + cnt_fp)
    recall = cnt_tp / (cnt_tp + cnt_fn)
    f1 = 2 * prec * recall / ((prec + recall))
    precs.append(prec)
    recalls.append(recall)
    f1s.append(f1)

print('precision = ' + str(np.mean(precs)))
print('recall = ' + str(np.mean(recalls)))
print('f1 = ' + str(np.mean(f1s)))






