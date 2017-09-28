import copy as cp
import subprocess
from model.model_config import DefaultConfig
import re

class MtEval_BLEU:

    def __init__(self, model_config):
        self.model_config = model_config
        self.template = ('<?xml version="1.0" encoding="UTF-8"?>\n' +
                    '<!DOCTYPE mteval SYSTEM "ftp://jaguar.ncsl.nist.gov/mt/resources/mteval-xml-v1.3.dtd">\n' +
                    '<mteval>\n' +
                    '<SET_LABEL setid="example_set" srclang="Arabic" trglang="English" refid="ref1" sysid="sample_system">\n' +
                    '<doc docid="doc1" genre="nw">\n' +
                    'CONTENT\n' +
                    '</doc>\n' +
                    '</SET_LABEL>\n' +
                    '</mteval>\n')

    def get_result(self, path_ref, path_src, path_tar):
        mteval_result = subprocess.check_output(['perl', self.model_config.mteval_script,
                                 '-r', path_ref,
                                 '-s', path_src,
                                 '-t', path_tar])
        m = re.search(b'BLEU score = (.+) for', mteval_result)
        return float(m.group(1))

    def get_bleu_from_decoderesult(self, sentence_complexs, sentence_simples, targets):
        path_ref = self.model_config.modeldir + '/mteval_reference.xml'
        path_src = self.model_config.modeldir + '/mteval_source.xml'
        path_tar = self.model_config.modeldir + '/mteval_target.xml'

        mteval_reference = open(path_ref, 'w', encoding='utf-8')
        mteval_source = open(path_src, 'w', encoding='utf-8')
        mteval_target = open(path_tar, 'w', encoding='utf-8')

        mteval_source.write(self.result2xml(sentence_complexs, 'srcset'))
        mteval_reference.write(self.result2xml(sentence_simples, 'refset'))
        mteval_target.write(self.result2xml(targets, 'tstset'))

        mteval_source.close()
        mteval_reference.close()
        mteval_target.close()

        return self.get_result(path_ref, path_src, path_tar)

    def get_bleu_from_rawresult(self, targets, path_gt_simple=None):
        if path_gt_simple is None:
            path_gt_simple = self.model_config.val_dataset_simple_folder + self.model_config.val_dataset_simple_file

        path_ref = self.model_config.modeldir + '/mteval_reference_real.xml'
        path_src = self.model_config.modeldir + '/mteval_source_real.xml'
        path_tar = self.model_config.modeldir + '/mteval_target_real.xml'

        mteval_reference = open(path_ref, 'w', encoding='utf-8')
        mteval_source = open(path_src, 'w', encoding='utf-8')
        mteval_target = open(path_tar, 'w', encoding='utf-8')

        mteval_source.write(self.txt2xml(self.model_config.val_dataset_complex, 'srcset'))
        mteval_reference.write(self.txt2xml(path_gt_simple, 'refset'))
        mteval_target.write(self.result2xml(targets, 'tstset'))
        mteval_source.close()
        mteval_reference.close()
        mteval_target.close()

        return self.get_result(path_ref, path_src, path_tar)

    def txt2xml(self, path, setlabel):
        sents = []
        for sent in open(path, encoding='utf-8'):
            sents.append(sent)
        return self.result2xml(sents, setlabel, join_split='')

    def result2xml(self, decode_result, setlabel, join_split=' '):
        tmp_output = ''
        for batch_i in range(len(decode_result)):
            tmp_line = join_split.join(decode_result[batch_i])
            tmp_line = '<p><seg id="%d"> %s </seg></p>' % (
            1 + batch_i, self.html_escape(tmp_line))
            tmp_output = '\n'.join([tmp_line, tmp_output])

        self.template_cp = cp.deepcopy(self.template)
        self.template_cp = self.template_cp.replace('SET_LABEL', setlabel)
        self.template_cp = self.template_cp.replace('CONTENT', tmp_output)
        return self.template_cp

    def html_escape(self, txt):
        txt = txt.replace('<','&lt;')
        txt = txt.replace('>', '&gt;')
        txt = txt.replace('', '')
        txt = txt.replace('&', '&amp;')
        txt = txt.replace('"', '&quot;')
        txt = txt.replace('\'', '&apos;')
        return txt

if __name__ == '__main__':
    bleu = MtEval_BLEU(DefaultConfig())
    dummy_result = [['a','b','c'],['e', 'f', 'g']]
    x = bleu.get_bleu_from_decoderesult(dummy_result, dummy_result, dummy_result)
    print(x)

