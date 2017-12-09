"""Deprecated: Move to java."""

from os import listdir
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import spacy


nlp = spacy.load('en')
titles = set()

def extract_simpe_titles(path='/Volumes/Storage/wiki/text_simp/AA/'):
    files = listdir(path)
    output = defaultdict(list)
    for file in files:
        start_doc = False
        cur_title = None
        for line in open('/Volumes/Storage/wiki/text_simp/AA/' + file, encoding='utf-8'):
            if line[:4] == '<doc':
                cur_title = line[line.index('title=')+7:line.rindex("\">")]
                if cur_title[:9] != 'Category:':
                    titles.add(cur_title)
                    start_doc = True
            elif line[:6] == '</doc>':
                # print('Finish Simp Wiki %s.' % cur_title)
                start_doc = False
                cur_title = None
            else:
                if start_doc and cur_title is not None:
                    line = line.replace(' .', ' . ').strip()
                    nline = sent_tokenize(
                        ' '.join([w.text for w in list(nlp(line))]))
                    output[cur_title].extend(nline)
                # else:
                #     print('Miss line\n%s' % line)
    print('Total Titles:%s' % len(titles))
    f = open('/Volumes/Storage/wiki/text_simp.txt', 'w', encoding='utf-8')
    for tmp_title in output:
        f.write('<TITLE=%s>' % tmp_title)
        f.write('\n')
        for sent in output[tmp_title]:
            f.write(sent)
            f.write('\n')
        f.write('</TITLE>')
        f.write('\n')
    f.close()


def extract_comp_titles(path='/Volumes/Storage/wiki/text_comp/AA/'):
    files = listdir(path)
    output = defaultdict(list)
    processed_title = set()
    for file in files:
        start_doc = False
        cur_title = None
        for line in open('/Volumes/Storage/wiki/text_comp/AA/' + file, encoding='utf-8'):
            if line[:4] == '<doc':
                try:
                    cur_title = line[line.index('title=') + 7:line.rindex("\">")]
                except:
                    print(line)
                    continue
                # print('Finish Comp Wiki %s.' % cur_title)
                if cur_title in titles:
                    processed_title.add(cur_title)
                    start_doc = True
            elif line[:6] == '</doc>':
                start_doc = False
                cur_title = None
            else:
                if start_doc and cur_title is not None:
                    line = line.replace(' .', ' . ').strip()
                    output[cur_title].extend(sent_tokenize(
                        ' '.join([w.text for w in list(nlp(line))])))
    f = open('/Volumes/Storage/wiki/text_comp.txt', 'w', encoding='utf-8')
    print('Processed Title:%s' % len(processed_title))
    for tmp_title in output:
        f.write('<TITLE=%s>' % tmp_title)
        f.write('\n')
        for sent in output[tmp_title]:
            f.write(sent)
            f.write('\n')
        f.write('</TITLE>')
        f.write('\n')
    f.close()

def extract_comp_titles_forall(path='/Volumes/Storage/wiki/text_comp/AA/'):
    """Extract all lines data for comp used for lm."""
    f = open('/Volumes/Storage/wiki/text_comp_all_raw.txt', 'w', encoding='utf-8')

    # f_cache = open('/Volumes/Storage/wiki/text_comp_all_titles.txt', 'w', encoding='utf-8')
    # title_caches = set([title.strip() for title in f_cache.readlines()])

    files = listdir(path)
    output = defaultdict(list)
    title_cnt = 0
    for file in files:
        start_doc = False
        cur_title = None
        for line in open('/Volumes/Storage/wiki/text_comp/AA/' + file, encoding='utf-8'):
            if line[:4] == '<doc':
                try:
                    cur_title = line[line.index('title=') + 7:line.rindex("\">")]
                except:
                    continue
                start_doc = True
            elif line[:6] == '</doc>':
                if len(output) >= 1000:
                    title_cnt += 1
                    print('Processed %s.' % (title_cnt*1000))
                    for tmp_title in output:
                        for sent in output[tmp_title]:
                            f.write(sent)
                            f.write('\n')
                    f.flush()
                    output = defaultdict(list)
                start_doc = False
                cur_title = None
            else:
                if start_doc and cur_title is not None:
                    line = line.replace(' .', ' . ').strip()
                    output[cur_title].extend(sent_tokenize(
                        ' '.join([w for w in list(line.split())])))
    for tmp_title in output:
        for sent in output[tmp_title]:
            f.write(sent)
            f.write('\n')
    f.close()


if __name__ == '__main__':
    # extract_simpe_titles()
    # extract_comp_titles()
    extract_comp_titles_forall()