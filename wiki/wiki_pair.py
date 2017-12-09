"""Deprecated: Move to java."""

import re
from collections import defaultdict
import copy as cp
from nltk.corpus import stopwords
import numpy as np
import math


stopWords = set(stopwords.words('english'))

class WikiGenerator:
    def init_tfidf(self):
        lines_comp = open('/Volumes/Storage/wiki/text_comp_nertag.txt', encoding='utf-8').readlines()
        lines_simp = open('/Volumes/Storage/wiki/text_simp_nertag.txt', encoding='utf-8').readlines()
        lines = lines_comp + lines_simp

        titles = []
        title = None
        tf = defaultdict(int)
        idf = defaultdict(set)
        cnt_word = 0
        cnt_title = 0
        for line_id, line in enumerate(lines):
            if line[:len('< TITLE = ')] == '< TITLE = ':
                title = re.search('< TITLE = ([^>]+)>', line).group(1)[:-1]
                title = title.replace('_TAG', '')
                titles.append(title)
                cnt_title += 1
            else:
                for word in line.strip().split():
                    word = word.replace('_TAG', '')
                    tf[word] += 1
                    cnt_word += 1
                    idf[word].add(title)
            if line_id % 100000 == 0:
                print('Processed tfidf %s' % line_id)
        self.tfidf = {}
        for word in tf:
            val = math.log(1 + tf[word]) * math.log(cnt_title / len(idf[word]))
            self.tfidf[word] = val
        print('Finished process tfidf.')

    def generate_doc(self):
        self.init_tfidf()
        f_comp = open('/Volumes/Storage/wiki/text_comp_nertag.txt', encoding='utf-8')
        title = None
        mapper_comp = defaultdict(list)
        append_before = False
        for line in f_comp:
            line = line.strip()
            if line[:len('< TITLE = ')] == '< TITLE = ':
                title = re.search('< TITLE = ([^>]+)>', line).group(1)[:-1]
                title = title.replace('_TAG', '')
                append_before = False
            else:
                if title is not None and len(line) > 0:
                    if append_before:
                        mapper_comp[title][-1] += (" " + line)
                    else:
                        mapper_comp[title].append(line)
                    append_before = len(line) >= 1 and line[-1] != '.' and line[-1] != '?' and line[-1] != '!'
        print('Load comp lines.')

        f_simp = open('/Volumes/Storage/wiki/text_simp_nertag.txt', encoding='utf-8')
        nlines_simp, nlines_comp, nlines_scores, nlines_title  = [], [], [], []
        title = None
        title_cnt = 0
        cands_simp = []
        append_before = False
        for line in f_simp:
            line = line.strip()
            if line[:len('< TITLE = ')] == '< TITLE = ':
                cands_comp = mapper_comp[title]
                if cands_simp and cands_comp:
                    temp_comp, temp_simp, temp_score = self.find_pairs(cands_simp, cands_comp)
                    assert len(temp_simp) == len(temp_comp) and len(temp_comp) == len(temp_score)
                    nlines_comp.extend(temp_comp)
                    nlines_simp.extend(temp_simp)
                    nlines_scores.extend(temp_score)
                    nlines_title.extend([title for _ in range(len(temp_score))])

                    title_cnt += 1
                    if title_cnt % 1000 == 0:
                        print('Processed %s titles.' % title_cnt)
                # For next title
                cands_simp = []
                append_before = False
                title = re.search('< TITLE = ([^>]+)>', line).group(1)[:-1]
                title = title.replace('_TAG', '')

                if title not in mapper_comp or len(mapper_comp[title]) == 0:
                    title = None
            elif title is not None and len(mapper_comp[title]) > 0 and len(line) > 0:
                if append_before:
                    cands_simp[-1] += (" " + line)
                else:
                    cands_simp.append(line)
                append_before = len(line) >= 1 and line[-1] != '.' and line[-1] != '?' and line[-1] != '!'

        f = open('/Volumes/Storage/wiki/text_pairs.txt', 'w')
        for i in range(len(nlines_scores)):
            f.write(nlines_comp[i])
            f.write('\n')
            f.write(nlines_simp[i])
            f.write('\n')
            f.write(nlines_title[i])
            f.write('\n')
            f.write(str(nlines_scores[i]))
            f.write('\n')
            f.write('=========================')
            f.write('\n')
        f.close()
        f_simp = open('/Volumes/Storage/wiki/text_simp_pairs.txt', 'w')
        f_comp = open('/Volumes/Storage/wiki/text_comp_pairs.txt', 'w')
        f_score = open('/Volumes/Storage/wiki/text_score_pairs.txt', 'w')
        f_title = open('/Volumes/Storage/wiki/text_title_pairs.txt', 'w')
        f_simp.write('\n'.join(nlines_simp))
        f_comp.write('\n'.join(nlines_comp))
        f_title.write('\n'.join(nlines_title))
        f_score.write('\n'.join(([str(s) for s in nlines_scores])))
        f_simp.close()
        f_comp.close()
        f_score.close()
        f_title.close()

    def find_pairs(self, cands_simp, cands_comp):
        tab = [[0 for _ in range(len(cands_comp))] for _ in range(len(cands_simp))]
        for id_simp in range(len(cands_simp)):
            for id_comp in range(len(cands_comp)):
                tab[id_simp][id_comp] = self.get_score(
                    cands_simp[id_simp], cands_comp[id_comp])

        assign_comp, assign_simp, assign_score = [], [], []
        post_comps, post_simps = [], []
        for id_simp in range(len(cands_simp)):
            id_comp = np.argmax(tab[id_simp])
            score = tab[id_simp][id_comp]
            if id_comp in assign_comp:
                post_simps.append(id_simp)
                post_comps.append(id_comp)

                id_del = assign_comp.index(id_comp)
                del assign_comp[id_del]
                del assign_simp[id_del]
                del assign_score[id_del]
            else:
                assign_comp.append(id_comp)
                assign_simp.append(id_simp)
                assign_score.append(score)

        if post_simps and False:
            post_simps = [_ for _ in set(post_simps)]
            post_comps = [_ for _ in (set(range(len(cands_comp)))-set(assign_comp))]
            if post_simps and post_comps:
                rare_comps = len(post_comps) < len(post_simps)
                print('Post Process with matrix %sX%s' % (len(post_simps), len(post_comps)))

                self.max_score = 0
                self.best_assignment = None
                self.best_scores = None
                self.find_pairs_recursive(0, [], [], [], cands_simp, cands_comp, post_simps, post_comps, tab, rare_comps)

                assign_simp.extend(self.best_assignment[0])
                assign_comp.extend(self.best_assignment[1])
                assign_score.extend(self.best_scores)

        # Get Concrete lines
        nlines_simp, nlines_comp, scores = [], [], []
        assert len(assign_comp) == len(assign_simp) and len(assign_score) == len(assign_comp)
        for i in range(len(assign_score)):
            if assign_score[i] == 0:
                continue

            nlines_simp.append(cands_simp[assign_simp[i]].replace('_TAG', ''))
            nlines_comp.append(cands_comp[assign_comp[i]].replace('_TAG', ''))
            scores.append(assign_score[i])
        return nlines_comp, nlines_simp, scores

    def find_pairs_recursive(
            self, post_id_simp, assignment_simp, assignment_comp, scores,
            cands_simp, cands_comp, post_simps, post_comps, tab, rare_comps):
        if post_id_simp == len(post_simps) or rare_comps:
            sum_score = sum(scores)
            if sum_score > self.max_score:
                self.max_score = sum_score
                self.best_assignment = (cp.deepcopy(assignment_simp), cp.deepcopy(assignment_comp))
                self.best_scores = cp.deepcopy(scores)
            return

        assignment_tar_set = set(assignment_comp)
        for id_comp in post_comps:
            if id_comp in assignment_tar_set:
                continue
            id_simp = post_simps[post_id_simp]
            score = tab[id_simp][id_comp]
            assignment_simp.append(id_simp)
            assignment_comp.append(id_comp)
            scores.append(score)
            self.find_pairs_recursive(
                post_id_simp + 1, assignment_simp, assignment_comp, scores,
                cands_simp, cands_comp, post_simps, post_comps, tab, rare_comps)
            del assignment_simp[-1]
            del assignment_comp[-1]
            del scores[-1]

    def get_score(self, cand_simp, cand_comp):
        set_simp = set(cand_simp.split()) - stopWords
        set_comp = set(cand_comp.split()) - stopWords
        shared_words = set_simp & set_comp
        score = 0
        for shared_word in shared_words:
            is_ner = shared_word[:len('_TAG')] == '_TAG'
            shared_word = shared_word.replace('_TAG', '')
            if len(shared_word) == 0:
                continue
            score += 3 * self.tfidf[shared_word] if is_ner else self.tfidf[shared_word]
        return score

    def remove_dup(self):
        def find_sim(query, collections):
            scores = [-1 for _ in range(len(collections))]
            query_set = set(query)
            for coll_id, collection in enumerate(collections):
                coll_set = set(collection)
                score = len(query_set & coll_set) / len(query_set | coll_set)
                scores[coll_id] = score
            return np.argmax(scores)

        path_val_comp = '/Users/zhaosanqiang916/git/text_simplification_data/val/wiki.full.aner.valid.src'
        path_val_simp = '/Users/zhaosanqiang916/git/text_simplification_data/val/wiki.full.aner.valid.dst'
        path_test_comp = '/Users/zhaosanqiang916/git/text_simplification_data/test/wiki.full.aner.test.src'
        path_test_simp = '/Users/zhaosanqiang916/git/text_simplification_data/test/wiki.full.aner.test.dst'

        res_comp = [l.strip().split() for l in open(path_val_comp, encoding='utf-8').readlines()] + [
            l.strip() for l in open(path_test_comp, encoding='utf-8').readlines()]
        res_simp = [l.strip().split() for l in open(path_val_simp, encoding='utf-8').readlines()] + [
            l.strip() for l in open(path_test_simp, encoding='utf-8').readlines()]

        lines_simp = [l.strip().split() for l in
                      open('/Volumes/Storage/wiki/text_simp_pairs.txt', encoding='utf-8').readlines()]
        lines_comp = [l.strip().split() for l in
                      open('/Volumes/Storage/wiki/text_comp_pairs.txt', encoding='utf-8').readlines()]
        lines_score =  open('/Volumes/Storage/wiki/text_score_pairs.txt', encoding='utf-8').readlines()
        lines_title = open('/Volumes/Storage/wiki/text_title_pairs.txt', encoding='utf-8').readlines()
        removed_ids = set()
        for i in range(len(res_comp)):
            print('Processed %s.' % i)
            id_comp = find_sim(res_comp[i], lines_comp)
            removed_ids.add(id_comp)
            id_simp = find_sim(res_simp[i], lines_simp)
            removed_ids.add(id_simp)

        print('removed %s sample.' % len(removed_ids))
        nlines_simp, nlines_comp, nlines_score, nlines_title = [], [], [], []
        for i in range(len(lines_score)):
            if i not in removed_ids:
                nlines_comp.append(' '.join(lines_comp[i]))
                nlines_simp.append(' '.join(lines_simp[i]))
                nlines_title.append(lines_title[i].strip())
                nlines_score.append(lines_score[i].strip())
        f_simp = open('/Volumes/Storage/wiki/text_simp_pairs.dup.txt', 'w')
        f_comp = open('/Volumes/Storage/wiki/text_comp_pairs.dup.txt', 'w')
        f_score = open('/Volumes/Storage/wiki/text_score_pairs.dup.txt', 'w')
        f_title = open('/Volumes/Storage/wiki/text_title_pairs.dup.txt', 'w')
        f_simp.write('\n'.join(nlines_simp))
        f_comp.write('\n'.join(nlines_comp))
        f_title.write('\n'.join(nlines_title))
        f_score.write('\n'.join(([str(s) for s in nlines_score])))
        f_simp.close()
        f_comp.close()
        f_score.close()
        f_title.close()


if __name__ == '__main__':
    wikigen = WikiGenerator()
    # wikigen.generate_doc()
    wikigen.remove_dup()
    # wikigen.find_pairs(
    #     ['xx bb xx' for _ in range(100)],
    #     ['xx bb dd' for _ in range(100)])
    # print('x')