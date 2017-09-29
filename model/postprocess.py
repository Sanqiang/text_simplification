import numpy as np
from util import constant
from scipy.spatial.distance import cosine
import copy as cp


class PostProcess:
    def __init__(self, model_config, data):
        self.model_config = model_config
        self.data = data

    def replace_unk_by_emb(self, encoder_words, encoder_embs, decoder_outputs, decoder_targets):
        batch_size = np.shape(decoder_targets)[0]

        ndecoder_targets = cp.deepcopy(decoder_targets)
        for batch_i in range(batch_size):
            for len_i in range(len(decoder_targets[batch_i])):
                target = decoder_targets[batch_i][len_i]
                if target == constant.SYMBOL_UNK or target == constant.SYMBOL_NUM:
                    query = decoder_outputs[batch_i, len_i, :]
                    word_exclude = set(ndecoder_targets[batch_i])
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
                if target == constant.SYMBOL_UNK or target == constant.SYMBOL_NUM:
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
