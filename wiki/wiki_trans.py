"""The code provides high-quality wiki data set as well as for trans data"""
class WikiTrans:
    def __init__(self, path_map, path_comp, path_simp, npath_comp, npath_simp):
        self.path_map = path_map
        self.path_comp = path_comp
        self.path_simp = path_simp
        self.npath_comp = npath_comp
        self.npath_simp = npath_simp

    def isvalid(self, sent):
        if len(sent) <= 3:
            return False
        if sent[-1] != '.' and sent[-1] != '!' and sent[-1] != '?':
            return False
        return True

    def filter(self):
        """Valid filter out noisy data"""
        f_comp = open(self.path_comp)
        f_simp = open(self.path_simp)
        f_map = open(self.path_map)
        f_comp2 = open(self.npath_comp, 'w')
        f_simp2 = open(self.npath_simp, 'w')
        lines_comp = f_comp.readlines()
        lines_simp = f_simp.readlines()
        lines_map = f_map.readlines()
        nlines_comp, nlines_simp = [], []
        assert len(lines_comp) == len(lines_simp)
        for i in range(len(lines_comp)):
            line_comp = lines_comp[i]
            line_simp = lines_simp[i]
            if self.isvalid(line_comp.split()) and self.isvalid(line_simp.split()):
                nlines_comp.append(line_comp)
                nlines_simp.append(line_simp)
            elif lines_map[i].strip():
                print(line_comp, line_simp, lines_map[i].strip(), '\n\n')

            if len(nlines_comp) >= 10000:
                f_comp2.write(''.join(nlines_comp))
                f_simp2.write(''.join(nlines_simp))
                f_comp2.flush()
                f_simp2.flush()
                nlines_simp.clear()
                nlines_comp.clear()
        f_comp2.write(''.join(nlines_comp))
        f_simp2.write(''.join(nlines_simp))
        f_comp2.close()
        f_simp2.close()
        f_comp.close()
        f_simp.close()

    def filter_ner(self, path_ner, npath_ner, npath_comp):
        import re
        f_compn = open(npath_comp, 'w')
        f_nern = open(npath_ner, 'w')
        f_comp = open(self.npath_comp, encoding='utf-8')
        f_ner = open(path_ner, encoding='utf-8')
        lines_comp = f_comp.readlines()
        lines_ner = f_ner.readlines()
        nlines_comp, nlines_ner = [], []
        for i in range(len(lines_comp)):
            is_valid = True
            line_comp = lines_comp[i]
            line_ner = lines_ner[i]
            wrong_ners = re.findall('((LOCATION|NUMBER|ORGANIZATION|PERSON)[^@][0-9])', line_ner)
            for wrong_ner in wrong_ners:
                m = re.match('(LOCATION|NUMBER|ORGANIZATION|PERSON)[^@]([0-9])', wrong_ner[0])
                if m is not None:
                    ner = m.group(1)
                    num = m.group(2)
                    ner = ner + '@' + num
                    if ner in line_comp:
                        line_ner = line_ner.replace(wrong_ner[0], ner)
                    else:
                        is_valid = False
            if is_valid:
                nlines_comp.append(line_comp)
                nlines_ner.append(line_ner)
            if len(nlines_ner) >= 10000:
                f_compn.write(''.join(nlines_comp))
                f_compn.flush()
                f_nern.write(''.join(nlines_ner))
                f_nern.flush()
                nlines_comp.clear()
                nlines_ner.clear()
        f_compn.write(''.join(nlines_comp))
        f_compn.close()
        f_nern.write(''.join(nlines_ner))
        f_nern.close()



if __name__ == '__main__':
    wiki = WikiTrans(
        '/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src.rules',
        '/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src',
        '/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.dst',
        '/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilarge2/src.txt',
        '/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilarge2/dst.txt')
    # wiki.filter()
    wiki.filter_ner('/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilarge2/trans/ndst.txt',
                    '/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilarge2/dst2.txt',
                    '/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilarge2/src2.txt')