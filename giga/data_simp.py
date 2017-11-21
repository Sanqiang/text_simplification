"""Deprecated: we found title files simp not work at all."""
from model.lm import GoogleLM
from model.ppdb import PPDB
from os import listdir
from model.model_config import WikiDressLargeTrainConfig
import os


class GigaSimp:
    def __init__(self):
        self.ppdb = PPDB(WikiDressLargeTrainConfig())
        self.google_lm = GoogleLM()

    def process(self):
        base = '/Volumes/Storage/giga_processed/'
        files = listdir(base)
        for i, file in enumerate(files):
            path = base + file
            if path[:-len('.title')] == '.title':
                npath = path + '.simp'
                if not os.path.exists(npath):
                    self.process_file(path)

    def process_file(self, path='/Volumes/Storage/giga_processed/afp_eng_199405.title'):
        text = open(path).readlines()
        nlines = []
        for line in text:
            line += ' .'
            prev_weight = self.google_lm.get_weight(line)

            best_line = line
            best_weight = prev_weight
            for cand_words in self.ppdb.rules:
                if len(cand_words) <= 1:
                    continue
                if cand_words in line:
                    targ_words = []
                    for tag in self.ppdb.rules[cand_words]:
                        for targ_word in self.ppdb.rules[cand_words][tag]:
                            targ_words.append(targ_word)

                    for targ_word in targ_words:
                        nline = line.replace(cand_words, targ_word)
                        nweight = self.google_lm.get_weight(nline)
                        if nweight < best_weight:
                            best_weight = nweight
                            best_line = nline
                    line = best_line
            nlines.append(line)
        f = open(path+'.simp', 'w')
        f.write('\n'.join(nlines))
        f.close()

if __name__ == '__main__':
    gs = GigaSimp()
    gs.process_file()



