import gzip
import re
from os import listdir


class Checker:
    DOC_START = 1
    DOC_END = 2
    HEADLINE_START = 3
    HEADLINE_END = 4
    TEXT_START = 5
    TEXT_END = 6
    Others = 0

class Giga:
    def _check_mark(self, line):
        self.text = False
        if len(line) > 0:
            if line[:4] == '<DOC' and line[line.index('type')+6:line.index('type')+11] == 'story':
                self.check_doc = True
            elif line[:4] == '<DOC' and line[line.index('type')+6:line.index('type')+11] != 'story':
                self.check_doc = False
            elif line == '<HEADLINE>':
                self.check_headline = True
            elif line == '</HEADLINE>':
                self.check_headline = False
            elif line == '<TEXT>':
                self.check_text = True
            elif line == '</TEXT>':
                self.check_text = False
            elif line[0] == '(':
                self.text = True

    def parse_text(self, line):
        line = line.strip()
        words = [w[:-1] for w in re.findall('[^)^ ]+\)', line)]
        return ' '.join(words)

    def process_file(self, path='/Volumes/Storage/giga/afp_eng_199405.xml.gz'):
        """Get the dataset for single file."""
        name = path[path.index('giga')+5: path.index('.xml')]
        titles, texts = [], []
        with gzip.open(path, 'r') as f:
            self.check_doc = False
            self.check_headline = False
            self.check_text = False
            title = None
            text = None
            for line in f.readlines():
                line = line.decode("utf-8").strip()
                self._check_mark(line)
                if self.check_doc and self.check_headline and self.text:
                    title = self.parse_text(line)
                elif self.check_doc and self.check_text and self.text:
                    text = self.parse_text(line)
                    if title is not None and text is not None:
                        # Only get first line of text
                        titles.append(title)
                        texts.append(text)
                        title = None
                        text = None
        f_title = open(
            '/Volumes/Storage/giga_processed/'+name+'.title', 'w', encoding='utf-8')
        f_title.write('\n'.join(titles))
        f_title.close()
        f_text = open(
            '/Volumes/Storage/giga_processed/'+name+'.text', 'w', encoding='utf-8')
        f_text.write('\n'.join(texts))
        f_text.close()

    def process(self):
        """Get the dataset."""
        base = '/Volumes/Storage/giga/'
        files = listdir(base)
        for i, file in enumerate(files):
            path = base + file
            try:
                self.process_file(path)
            except:
                print('Except file %s' % path)
            print('Finished file %s:%s.' % (i, file))

    def aggregate(self):
        """Aggregate the dataset into one and filtering"""
        base = '/Volumes/Storage/giga_processed/'
        ntexts = []
        ntitles = []
        files = listdir(base)
        for i, file in enumerate(files):
            if file[-len('.title'):] == '.title':
                text_path = base + file[:-len('.title')] + '.text'
                title_path = base + file

                #Aggregate one file
                print('aggregate %s.' % title_path)
                texts_tmp = open(text_path, encoding='utf-8').readlines()
                titles_tmp = open(title_path, encoding='utf-8').readlines()
                assert len(texts_tmp) == len(titles_tmp)
                for i in range(len(texts_tmp)):
                    text_tmp = texts_tmp[i].strip()
                    title_tmp = titles_tmp[i].strip() + ' .'
                    text_tmp_set = set(text_tmp.split())
                    title_tmp_set = set(title_tmp.split())
                    if len(text_tmp_set & title_tmp_set) >= 10 and len(title_tmp_set) >= 10:
                        ntexts.append(text_tmp)
                        ntitles.append(title_tmp)
        f_simp = open('/Volumes/Storage/giga10.simp', 'w', encoding='utf-8')
        f_comp = open('/Volumes/Storage/giga10.comp', 'w', encoding='utf-8')
        f_simp.write('\n'.join(ntitles))
        f_comp.write('\n'.join(ntexts))
        f_simp.close()
        f_comp.close()


if __name__ == '__main__':
    # Giga().process()
    Giga().aggregate()