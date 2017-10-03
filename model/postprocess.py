import numpy as np
from util import constant
from scipy.spatial.distance import cosine
from scipy import stats
import copy as cp


class PostProcess:
    def __init__(self, model_config, data):
        self.model_config = model_config
        self.data = data

    def replace_ner(self, decoder_targets, mapper):
        batch_size = np.shape(decoder_targets)[0]
        ndecoder_targets = cp.deepcopy(decoder_targets)
        for batch_i in range(batch_size):
            for len_i in range(len(decoder_targets[batch_i])):
                word = decoder_targets[batch_i][len_i]
                if word in mapper[batch_i]:
                    ndecoder_targets[batch_i][len_i] = mapper[batch_i][word]
        return ndecoder_targets

    def replace_others(self, decoder_targets):
        batch_size = np.shape(decoder_targets)[0]
        ndecoder_targets = cp.deepcopy(decoder_targets)
        for batch_i in range(batch_size):
            for len_i in range(len(decoder_targets[batch_i])):
                word = decoder_targets[batch_i][len_i]
                # Nothing so far
        return ndecoder_targets

    def replace_unk_by_attn(self, encoder_words, attn_dists, decoder_targets):
        batch_size = np.shape(decoder_targets)[0]
        decoder_targets[0][3] = constant.SYMBOL_UNK
        decoder_targets[1][3] = constant.SYMBOL_UNK

        ndecoder_targets = []
        for batch_i in range(batch_size):
            if len(np.shape(attn_dists)) == 2:
                ndecoder_target = self.replace_unk_by_attn_onestep(
                    encoder_words[batch_i], attn_dists[batch_i], decoder_targets[batch_i])
                ndecoder_targets.append(ndecoder_target)
            elif len(np.shape(attn_dists)) == 3:
                num_heads = np.shape(attn_dists)[1]
                ndecoder_target_cands = []
                for head_i in range(num_heads):
                    ndecoder_target_cand = self.replace_unk_by_attn_onestep(
                        encoder_words[batch_i], attn_dists[batch_i][head_i], decoder_targets[batch_i])
                    ndecoder_target_cands.append(ndecoder_target_cand)
                ndecoder_target_cands = np.array(ndecoder_target_cands)
                # Majority vote for all heads result
                mv_result = stats.mode(ndecoder_target_cands)
                ndecoder_targets.append(list(mv_result[0]))
        return ndecoder_targets

    def replace_unk_by_attn_onestep(self, encoder_word, attn_dist, decoder_target):
        ndecoder_target = cp.deepcopy(decoder_target)
        for len_i in range(len(decoder_target)):
            target = decoder_target[len_i]
            if target == constant.SYMBOL_UNK:
                nword = encoder_word[attn_dist[len_i]]
                ndecoder_target[len_i] = nword
        return ndecoder_target

    def replace_unk_by_emb(self, encoder_words, encoder_embs, decoder_outputs, decoder_targets):
        batch_size = np.shape(decoder_targets)[0]

        ndecoder_targets = cp.deepcopy(decoder_targets)
        for batch_i in range(batch_size):
            for len_i in range(len(decoder_targets[batch_i])):
                target = decoder_targets[batch_i][len_i]
                if target == constant.SYMBOL_UNK:
                    query = decoder_outputs[batch_i, len_i, :]
                    word_exclude = set()
                    # word_exclude = set(ndecoder_targets[batch_i])
                    word_exclude.update([
                        constant.SYMBOL_START, constant.SYMBOL_END, constant.SYMBOL_UNK,
                        constant.SYMBOL_PAD, constant.SYMBOL_GO])
                    dists = [99999 for _ in range(len(encoder_words[batch_i]))]
                    replace = False
                    for loop_i in range(len(encoder_words[batch_i])):
                        if encoder_words[batch_i][loop_i] in word_exclude:
                            continue
                        emb = encoder_embs[batch_i, loop_i, :]
                        dists[loop_i] = cosine(query, emb)
                        replace = True

                    target_idx = -1
                    if replace:
                        target_idx = np.argmin(dists)
                    else:
                        word_exclude = set()
                        word_exclude.update([constant.SYMBOL_START, constant.SYMBOL_END, constant.SYMBOL_UNK,
                                             constant.SYMBOL_PAD, constant.SYMBOL_GO])
                        dists = [99999 for _ in range(len(encoder_words[batch_i]))]
                        for loop_i in range(len(encoder_words[batch_i])):
                            if encoder_words[batch_i][loop_i] in word_exclude:
                                continue
                            emb = encoder_embs[batch_i, loop_i, :]
                            dists[loop_i] = cosine(query, emb)
                            replace = True
                        if replace:
                            target_idx = np.argmin(dists)
                    if target_idx >= 0:
                        target_word = encoder_words[batch_i][target_idx]
                        ndecoder_targets[batch_i][len_i] = target_word

        return ndecoder_targets

    def replace_unk_by_cnt(self, encoder_words, decoder_targets, window=5):
        batch_size = np.shape(decoder_targets)[0]

        ndecoder_targets = cp.deepcopy(decoder_targets)
        for batch_i in range(batch_size):
            for len_i in range(len(decoder_targets[batch_i])):
                target = decoder_targets[batch_i][len_i]
                if target == constant.SYMBOL_UNK:
                    word_cands = set(encoder_words[batch_i]) - set(ndecoder_targets[batch_i])
                    exclude_word = set([
                        constant.SYMBOL_START, constant.SYMBOL_END, constant.SYMBOL_UNK,
                        constant.SYMBOL_PAD, constant.SYMBOL_GO])
                    word_cands -= exclude_word
                    word_cands = list(word_cands)
                    word_cands_point = [0 for _ in range(len(word_cands))]

                    target_len = len(decoder_targets[batch_i])
                    words = set()
                    for loop_i in range(1, window+1):
                        if len_i - loop_i >= 0:
                            word = decoder_targets[batch_i][len_i - loop_i]
                            words.add(word)
                        if len_i + loop_i < target_len:
                            word = decoder_targets[batch_i][len_i + loop_i]
                            words.add(word)

                    for i, word_cand in enumerate(word_cands):
                        word_cand_id = encoder_words[batch_i].index(word_cand)
                        for loop_i in range(1, window + 1):
                            if word_cand_id - loop_i >= 0:
                                word = encoder_words[batch_i][word_cand_id - loop_i]
                                if word in words:
                                    word_cands_point[i] += 1
                            if word_cand_id + loop_i < target_len:
                                word = encoder_words[batch_i][word_cand_id + loop_i]
                                if word in words:
                                    word_cands_point[i] += 1

                    ndecoder_targets[batch_i][len_i] = word_cands[np.argmax(word_cands_point)]

        return ndecoder_targets

if __name__ == '__main__':
    sents = ['the term #quot# union council #quot# may be used for cities that are part of their cities .'.split()]
    sents = PostProcess(None, None).replace_others(sents)
    print(sents)