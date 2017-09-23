import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from util import constant


class Metric:
    def __init__(self, model_config, data):
        self.model_config = model_config
        self.data = data


    def bleu(self, sentence_simple_input, sentence_complex_input):
        batch_size = np.shape(sentence_simple_input)[0]
        bleus = []

        for batch_i in range(batch_size):
            sent_simple = [self.data.vocab_simple.describe(wid)
                           for wid in sentence_simple_input[batch_i, :]
                           if wid != self.data.vocab_simple.encode(constant.SYMBOL_PAD)]
            sent_complex = [self.data.vocab_complex.describe(wid)
                           for wid in sentence_complex_input[batch_i, :]
                            if wid != self.data.vocab_complex.encode(constant.SYMBOL_PAD)]
            bleu = sentence_bleu(sent_simple, sent_complex)
            bleus.append(bleu)

        return np.array(bleus, dtype=np.float32)