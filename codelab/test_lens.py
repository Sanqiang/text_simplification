import pyphen
dic = pyphen.Pyphen(lang='en')

def get_cnt_syl(word):
    return len(dic.inserted(word).split('-'))


output_path = '/zfs1/hdaqing/saz31/text_simplification/dress_final_ffn_cl0/result/eightref_test_ppdbe_v2/joshua_target_1159549.txt'
lines = open(output_path).readlines()

cnt_sent = len(lines)
cnt_word, cnt_syl = 0, 0
for line in lines:
    words = line.strip().lower().split()
    cnt_word += len(words)
    for word in words:
        cnt_syl += get_cnt_syl(word)

print('slen = %s' % str(cnt_word / cnt_sent))
print('wlen = %s' % str(cnt_syl / cnt_word))