"""Prepare NER-parsed dataset.
   Preruned locally, so don't need to install lib in server."""
from nltk.tag import StanfordNERTagger, SennaNERTagger
import copy as cp
from os.path import exists
from os import makedirs
import os
from util import map_util
import numpy as np
from nltk import word_tokenize

from model.model_config import get_path

class DataNERPrepareBase:

    def __init__(self, out_path='out.txt', map_path='map.txt'):
        self.st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')

        self.concat_tag_set = set(['PERSON', 'LOCATION', 'ORGANIZATION'])
        self.replace_tag_set = set(['PERSON', 'LOCATION', 'ORGANIZATION', 'NUMBER'])

    def process(self):
        for stage in ['eval', 'test']:
            # For wiki 8 ref
            if stage == 'eval':
                ori_complex_path = get_path(
                    '../text_simplification_data/val/tune.8turkers.tok.norm.raw')
                ori_simple_path = get_path(
                    '../text_simplification_data/val/tune.8turkers.tok.simp.raw')
                ori_reference_path = get_path(
                    '../text_simplification_data/val/tune.8turkers.organized.tsv')
                out_complex_path = get_path(
                    '../text_simplification_data/val/tune.8turkers.tok.norm.processed')
                out_simple_path = get_path(
                    '../text_simplification_data/val/tune.8turkers.tok.simp.processed')
                out_mapper_path = get_path(
                    '../text_simplification_data/val/tune.8turkers.tok.map')
                out_refernce_raw_path = get_path(
                    '../text_simplification_data/val/tune.8turkers.tok.turk.raw.')
                out_reference_path = get_path(
                    '../text_simplification_data/val/tune.8turkers.tok.turk.processed.')
            elif stage == 'test':
                ori_complex_path = get_path(
                    '../text_simplification_data/test/test.8turkers.tok.norm.raw')
                ori_simple_path = get_path(
                    '../text_simplification_data/test/test.8turkers.tok.simp.raw')
                ori_reference_path = get_path(
                    '../text_simplification_data/test/test.8turkers.organized.tsv')
                out_complex_path = get_path(
                    '../text_simplification_data/test/test.8turkers.tok.norm.processed')
                out_simple_path = get_path(
                    '../text_simplification_data/test/test.8turkers.tok.simp.processed')
                out_mapper_path = get_path(
                    '../text_simplification_data/test/test.8turkers.tok.map')
                out_refernce_raw_path = get_path(
                    '../text_simplification_data/test/test.8turkers.tok.turk.raw.')
                out_reference_path = get_path(
                    '../text_simplification_data/test/test.8turkers.tok.turk.processed.')

            complex_lines = open(ori_complex_path, encoding='utf-8').readlines()
            simple_lines = open(ori_simple_path, encoding='utf-8').readlines()
            refer_lines = open(ori_reference_path, encoding='utf-8').readlines()

            # Sentences in reference files are stacked together.
            refer_lines_arr = [[] for _ in range(8)]
            for lid, line in enumerate(refer_lines):
                blocks = line.split('\t')
                sent_complex = blocks[1]
                if (sent_complex.replace('(', '').replace(')', '').replace(' ', '').replace(
                        '\\', "").replace('\'', "").replace('"', "").replace('`', '').replace('/', '').replace(
                    '[', '').replace(']', '').strip() !=
                        complex_lines[lid].replace('-LRB-', '').replace('-RRB-', '').replace(' ', '').replace(
                            '\\', "").replace('\'', "").replace('"', "").replace('`', '').replace('/', '').strip()):
                    raise Exception('Data map error!')

                for rid in range(8):
                    refer_lines_arr[rid].append([self.replace_word(word)
                                                 for word in word_tokenize(blocks[2+rid]) if self.check_word(word)])

            complex_lines = [complex_line.split() for complex_line in complex_lines]
            simple_lines = [simple_line.split() for simple_line in simple_lines]
            complex_line_tags = self.concat_tag_sents(self.st.tag_sents(complex_lines))
            simple_line_tags = self.concat_tag_sents(self.st.tag_sents(simple_lines))
            refer_lines_arr_tags = [[] for _ in range(8)]
            for rid in range(8):
                refer_lines_arr_tags[rid] = self.concat_tag_sents(self.st.tag_sents(refer_lines_arr[rid]))

            ncomplex_lines = []
            nsimple_lines = []
            nrefer_lines = [[] for _ in range(8)]
            mappers = []
            for lid in range(len(complex_lines)):
                print('Process idx %d' % lid)
                complex_line = [p[0] for p in complex_line_tags[lid]]
                simple_line = [p[0] for p in simple_line_tags[lid]]

                # Get Mapper
                complex_line_tag = set([(p[0], p[1])
                                        for p in complex_line_tags[lid] if p[1] in self.replace_tag_set])
                simple_line_tag = set([(p[0], p[1])
                                       for p in simple_line_tags[lid] if p[1] in self.replace_tag_set])
                all_tag = complex_line_tag | simple_line_tag
                for rid in range(8):
                    all_tag |= set([(p[0], p[1])
                                    for p in refer_lines_arr_tags[rid][lid] if p[1] in self.replace_tag_set])

                mapper_tag = {}

                for tag_type in self.replace_tag_set:
                    mapper_idx = 1
                    for tag_pair in all_tag:
                        if tag_pair[1] == tag_type:
                            mapper_tag[tag_pair[0]] = tag_pair[1] + '@' + str(mapper_idx)
                            mapper_idx += 1

                # Update mapper
                mapper = {}
                mapper.update(mapper_tag)
                mappers.append(mapper)

                # Replace based on Mapper
                ncomplex_line = cp.deepcopy(complex_line)
                for wid, word in enumerate(complex_line):
                    if word in mapper:
                        ncomplex_line[wid] = mapper[word]
                ncomplex_lines.append(ncomplex_line)

                nsimple_line = cp.deepcopy(simple_line)
                for wid, word in enumerate(simple_line):
                    if word in mapper:
                        nsimple_line[wid] = mapper[word]
                nsimple_lines.append(nsimple_line)

                for rid in range(8):
                    tmp = cp.deepcopy(refer_lines_arr[rid][lid])
                    for wid, word in enumerate(refer_lines_arr[rid][lid]):
                        if word in mapper:
                            tmp[wid] = mapper[word]
                    nrefer_lines[rid].append(tmp)

            f_ncomplex = open(out_complex_path, 'w', encoding='utf-8')
            f_nsimple = open(out_simple_path, 'w', encoding='utf-8')
            for ncomplex_line in ncomplex_lines:
                f_ncomplex.write(' '.join(ncomplex_line))
                f_ncomplex.write('\n')
            f_ncomplex.close()
            for nsimple_line in nsimple_lines:
                f_nsimple.write(' '.join(nsimple_line))
                f_nsimple.write('\n')
            for rid in range(8):
                f_ref_tmp = open(out_reference_path + str(rid), 'w', encoding='utf-8')
                for tmp in nrefer_lines[rid]:
                    f_ref_tmp.write(' '.join(tmp))
                    f_ref_tmp.write('\n')
                f_ref_tmp.close()
            for rid in range(8):
                f_ref_tmp = open(out_refernce_raw_path + str(rid), 'w', encoding='utf-8')
                for tmp in refer_lines_arr[rid]:
                    f_ref_tmp.write(' '.join(tmp))
                    f_ref_tmp.write('\n')
                f_ref_tmp.close()

            map_util.dump_mappers(mappers, out_mapper_path)

    def process_dress(self):
        for stage in ['eval', 'test']:
            # For wiki dress
            if stage == 'eval':
                ori_complex_path = get_path(
                    '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.ori.valid.src')
                ori_simple_path = get_path(
                    '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.ori.valid.dst')
                out_complex_path = get_path(
                    '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.ori.valid.src.processed')
                out_simple_path = get_path(
                    '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.ori.valid.dst.processed')
                out_mapper_path = get_path(
                    '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.ori.valid.map')
            elif stage == 'test':
                ori_complex_path = get_path(
                    '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.ori.test.src')
                ori_simple_path = get_path(
                    '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.ori.test.dst')
                out_complex_path = get_path(
                    '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.ori.test.src.processed')
                out_simple_path = get_path(
                    '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.ori.test.dst.processed')
                out_mapper_path = get_path(
                    '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.ori.test.map')

            complex_lines = open(ori_complex_path, encoding='utf-8').readlines()
            simple_lines = open(ori_simple_path, encoding='utf-8').readlines()

            complex_lines = [complex_line.split() for complex_line in complex_lines]
            simple_lines = [simple_line.split() for simple_line in simple_lines]
            complex_line_tags = self.concat_tag_sents(self.st.tag_sents(complex_lines))
            simple_line_tags = self.concat_tag_sents(self.st.tag_sents(simple_lines))

            ncomplex_lines = []
            nsimple_lines = []
            mappers = []
            for lid in range(len(complex_lines)):
                print('Process idx %d' % lid)
                complex_line = complex_lines[lid]
                simple_line = simple_lines[lid]

                # Get Mapper
                complex_line_tag = set([(p[0], p[1])
                                        for p in complex_line_tags[lid] if p[1] in self.replace_tag_set])
                simple_line_tag = set([(p[0], p[1])
                                       for p in simple_line_tags[lid] if p[1] in self.replace_tag_set])
                all_tag = complex_line_tag | simple_line_tag

                mapper_tag = {}

                for tag_type in self.replace_tag_set:
                    mapper_idx = 1
                    for tag_pair in all_tag:
                        if tag_pair[1] == tag_type:
                            mapper_tag[tag_pair[0]] = tag_pair[1] + '@' + str(mapper_idx)
                            mapper_idx += 1

                # Update mapper
                mapper = {}
                mapper.update(mapper_tag)
                mappers.append(mapper)

                # Replace based on Mapper
                ncomplex_line = cp.deepcopy(complex_line)
                for wid, word in enumerate(complex_line):
                    if word in mapper:
                        ncomplex_line[wid] = mapper[word]
                ncomplex_lines.append(ncomplex_line)

                nsimple_line = cp.deepcopy(simple_line)
                for wid, word in enumerate(simple_line):
                    if word in mapper:
                        nsimple_line[wid] = mapper[word]
                nsimple_lines.append(nsimple_line)

            f_ncomplex = open(out_complex_path, 'w', encoding='utf-8')
            f_nsimple = open(out_simple_path, 'w', encoding='utf-8')
            for ncomplex_line in ncomplex_lines:
                f_ncomplex.write(' '.join(ncomplex_line))
                f_ncomplex.write('\n')
            f_ncomplex.close()
            for nsimple_line in nsimple_lines:
                f_nsimple.write(' '.join(nsimple_line))
                f_nsimple.write('\n')

            map_util.dump_mappers(mappers, out_mapper_path)

    def prepare_raw_data(self):
        """Get raw valid and test data."""
        # PWKP_108016 aggregate complex and simple sentneces together with line seperator.
        cand_path = get_path('../text_simplification_data/pwkp108016/PWKP_108016')
        cands = open(cand_path, encoding='utf-8').readlines()
        cand_complexs, cand_simples = [], []
        cand_complexs_l, cand_simples_l = [], []
        is_complex, is_simple = True, False
        for i, cand in enumerate(cands):
            print('Processed PWKP_108016 for line:%d' % i)
            cand = [self.replace_word(word)
                    for word in word_tokenize(cand) if self.check_word(word)]
            if not cand and is_simple:
                is_simple = False
                is_complex = True
            elif is_complex:
                cand_complexs.append(cand)
                cand_complexs_l.append([c.lower() for c in cand])
                is_simple = True
                is_complex = False
            elif is_simple:
                if len(cand_simples) == len(cand_complexs):
                    cand_simples[-1].extend(cand)
                    cand_simples_l[-1].extend([c.lower() for c in cand])
                elif len(cand_simples) < len(cand_complexs):
                    cand_simples.append(cand)
                    cand_simples_l.append([c.lower() for c in cand])
        assert len(cand_complexs) == len(cand_simples)

        for stage in ["eval", "test"]:
            if stage == 'eval':
                cased_complex_path = get_path(
                    '../text_simplification_data/val/tune.8turkers.tok.norm')
                cased_simple_path = get_path(
                    '../text_simplification_data/val/tune.8turkers.tok.simp')
                raw_complex_path = get_path(
                    '../text_simplification_data/val/tune.8turkers.tok.norm.raw')
                raw_simple_path = get_path(
                    '../text_simplification_data/val/tune.8turkers.tok.simp.raw')
            elif stage == 'test':
                cased_complex_path = get_path(
                    '../text_simplification_data/test/test.8turkers.tok.norm')
                cased_simple_path = get_path(
                    '../text_simplification_data/test/test.8turkers.tok.simp')
                raw_complex_path = get_path(
                    '../text_simplification_data/test/test.8turkers.tok.norm.raw')
                raw_simple_path = get_path(
                    '../text_simplification_data/test/test.8turkers.tok.simp.raw')

            output_complex = ''
            output_simple = ''

            qeury_complexs = open(cased_complex_path, encoding='utf-8').readlines()
            qeury_complexs = [sents.lower().split() for sents in qeury_complexs]
            query_simples = open(cased_simple_path, encoding='utf-8').readlines()
            query_simples = [sents.lower().split() for sents in query_simples]

            for i in range(len(query_simples)):
                qeury_complex = qeury_complexs[i]
                best_complexs = [len(set(qeury_complex) & set(cand_complex)) / len(set(qeury_complex) | set(cand_complex))
                                 for cand_complex in cand_complexs_l]
                best_id = np.argmax(best_complexs)

                # Hand corrected.
                if stage == 'eval' and i == 489:
                    best_id = 70941
                if stage == 'eval' and i == 345:
                    query_simple = query_simples[i]
                    best_simples = [
                        len(set(query_simple) & set(cand_simple)) / len(set(query_simple) | set(cand_simple))
                        for cand_simple in cand_simples_l]
                    best_id = np.argmax(best_simples)

                output_complex = '\n'.join([output_complex, ' '.join(cand_complexs[best_id])])
                output_simple = '\n'.join([output_simple, ' '.join(cand_simples[best_id])])

                print('Processed %d' % i)

            f_complex = open(raw_complex_path, 'w', encoding='utf-8')
            f_simple = open(raw_simple_path, 'w', encoding='utf-8')
            f_complex.write(output_complex.strip())
            f_complex.close()
            f_simple.write(output_simple.strip())
            f_simple.close()

    def concat_tag_sents(self, tag_sents):
        ntag_sents = []
        for tag_sent in tag_sents:
            ntag_sent = []
            last_tag = ''
            for id in range(len(tag_sent)):
                tag = tag_sent[id][1]
                word = tag_sent[id][0]
                if self.is_numeric(word):
                    tag = "NUMBER"

                if tag == last_tag and tag in self.concat_tag_set:
                    ntag_sent[-1][0] += ' ' + word
                else:
                    ntag_sent.append([word, tag])
                last_tag = tag
            ntag_sents.append(ntag_sent)
        return ntag_sents

    def replace_word(self, word):
        if word == '(':
            return '-LRB-'
        elif word == ')':
            return '-RRB-'
        return word

    def check_word(self, word):
        if word == '[' or word == ']' or word.strip() == '':
            return False
        return True

    def is_numeric(self, word):
        """Slow-version numeric checking."""
        try:
            float(word)
        except ValueError:
            return False
        return True



if __name__ == '__main__':
    data_ner = DataNERPrepareBase()
    # data_ner.process_dress()
    data_ner.process()
    # data_ner.prepare_raw_data()