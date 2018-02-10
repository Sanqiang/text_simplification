import copy as cp
import subprocess
from model.model_config import DefaultConfig
import re
import os
from util import constant

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

    """Get Result for script/mteval-v13a.pl."""

    def get_result_singleref(self, path_ref, path_src, path_tar):
        mteval_result = subprocess.check_output(['perl', self.model_config.mteval_script,
                                 '-r', path_ref,
                                 '-s', path_src,
                                 '-t', path_tar])
        m = re.search(b'BLEU score = (.+) for', mteval_result)
        try:
            result = float(m.group(1))
        except AttributeError:
            result = 0
        return result


    def get_bleu_from_decoderesult(self, step, sentence_complexs, sentence_simples, targets):
        path_ref = self.model_config.resultdir + '/mteval_reference_%s.xml' % step
        path_src = self.model_config.resultdir + '/mteval_source_%s.xml' % step
        path_tar = self.model_config.resultdir + '/mteval_target_%s.xml' % step

        mteval_reference = open(path_ref, 'w', encoding='utf-8')
        mteval_source = open(path_src, 'w', encoding='utf-8')
        mteval_target = open(path_tar, 'w', encoding='utf-8')

        mteval_source.write(self.result2xml(sentence_complexs, 'srcset'))
        mteval_reference.write(self.result2xml(sentence_simples, 'refset'))
        mteval_target.write(self.result2xml(targets, 'tstset'))

        mteval_source.close()
        mteval_reference.close()
        mteval_target.close()

        return self.get_result_singleref(path_ref, path_src, path_tar)

    def get_bleu_from_rawresult(self, step, targets, path_gt_simple=None, path_gt_complex=None):
        if path_gt_simple is None:
            path_gt_simple = self.model_config.val_dataset_simple_folder + self.model_config.val_dataset_simple_rawlines_file
        if path_gt_complex is None:
            path_gt_complex = self.model_config.val_dataset_complex_rawlines_file

        path_ref = self.model_config.resultdir + '/mteval_reference_real_%s.xml' % step
        path_src = self.model_config.resultdir + '/mteval_source_real_%s.xml' % step
        path_tar = self.model_config.resultdir + '/mteval_target_real_%s.xml' % step

        mteval_reference = open(path_ref, 'w', encoding='utf-8')
        mteval_source = open(path_src, 'w', encoding='utf-8')
        mteval_target = open(path_tar, 'w', encoding='utf-8')

        mteval_source.write(self.path2xml(path_gt_complex, 'srcset',
                                          lower_case=self.model_config.lower_case))
        mteval_reference.write(self.path2xml(path_gt_simple, 'refset',
                                             lower_case=self.model_config.lower_case))
        mteval_target.write(self.result2xml(targets, 'tstset'))
        mteval_source.close()
        mteval_reference.close()
        mteval_target.close()

        return self.get_result_singleref(path_ref, path_src, path_tar)

    def path2xml(self, path, setlabel, lower_case=False):
        sents = []
        for sent in open(path, encoding='utf-8'):
            if lower_case:
                sent = sent.lower()
            sent = sent.strip()
            sents.append([sent])
        return self.result2xml(sents, setlabel, join_split='')

    def result2xml(self, decode_result, setlabel, join_split=' '):
        texts = []
        for batch_i in range(len(decode_result)):
            text = join_split.join(decode_result[batch_i])
            texts.append(text)
        return self.text2xml(texts, setlabel)

    def text2xml(self, texts, setlabel):
        tmp_output = ''
        for batch_i in range(len(texts)):
            tmp_line = texts[batch_i]
            tmp_line = '<p><seg id="%d"> %s </seg></p>' % (
            1 + batch_i, self.html_escape(tmp_line))
            tmp_output = '\n'.join([tmp_line, tmp_output])

        self.template_cp = cp.deepcopy(self.template)
        self.template_cp = self.template_cp.replace('SET_LABEL', setlabel)
        self.template_cp = self.template_cp.replace('CONTENT', tmp_output)
        return self.template_cp.strip()

    def html_escape(self, txt):
        txt = txt.replace('<','#lt#')
        txt = txt.replace('>', '#rt#')
        txt = txt.replace('&', '#and#')
        txt = txt.replace('"', constant.SYMBOL_QUOTE)
        txt = txt.replace('\'\'', constant.SYMBOL_QUOTE)
        txt = txt.replace('\'', constant.SYMBOL_QUOTE)
        txt = txt.replace('``', constant.SYMBOL_QUOTE)
        txt = txt.replace('`', constant.SYMBOL_QUOTE)
        return txt.strip()

    """Get Result for script/multi-bleu.perl."""

    def get_bleu_from_decoderesult_multirefs(self, step, path_ref, targets, lowercase=False):
        path_tar = self.model_config.resultdir + '/multibleu_target_%s.txt' % step
        f = open(path_tar, 'w', encoding='utf-8')
        f.write(self.result2txt(targets, lowercase=lowercase))
        f.close()

        return self.get_result_multiref(path_ref, path_tar, lowercase=lowercase)

    def result2txt(self, sents, lowercase=False, join_split=' '):
        nsents = []
        for sent in sents:
            sent = join_split.join(sent)
            if lowercase:
                sent = sent.lower()
            sent = sent.strip()
            nsents.append(sent)

        nsents = '\n'.join(nsents)
        return nsents

    def get_result_multiref(self, path_ref, path_tar, lowercase):
        args = ' '.join([self.model_config.mteval_mul_script, path_ref, '<', path_tar])
        if lowercase:
            args = ' '.join([self.model_config.mteval_mul_script, '-lc', path_ref, '<', path_tar])

        pipe = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        mteval_result = pipe.communicate()

        m = re.search(b'BLEU = ([\d+\.]+)', mteval_result[0])
        try:
            result = float(m.group(1)) /100.0
        except AttributeError:
            result = 0

        return result

    """Get Result for joshua"""

    def get_bleu_from_joshua(self, step, path_dst, path_ref, targets):
        path_tar = self.model_config.resultdir + '/joshua_target_%s.txt' % step
        if not os.path.exists(path_tar):
            f = open(path_tar, 'w', encoding='utf-8')
            # joshua require lower case
            f.write(self.result2txt(targets, lowercase=True))
            f.close()

        if self.model_config.num_refs > 0:
            return self.get_result_joshua(path_ref, path_tar)
        else:
            return self.get_result_joshua_nonref(path_dst, path_tar)

    def get_result_joshua(self, path_ref, path_tar):
        args = ' '.join([self.model_config.joshua_script, path_tar, path_ref,
                         str(self.model_config.num_refs), self.model_config.joshua_class])

        pipe = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        mteval_result = pipe.communicate()

        m = re.search(b'BLEU = ([\d+\.]+)', mteval_result[0])

        try:
            result = float(m.group(1))
        except AttributeError:
            result = 0
        return result

    def get_result_joshua_nonref(self, path_ref, path_tar):
        args = ' '.join([self.model_config.joshua_script, path_tar, path_ref,
                         '1', self.model_config.joshua_class])

        pipe = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        mteval_result = pipe.communicate()

        m = re.search(b'BLEU = ([\d+\.]+)', mteval_result[0])

        try:
            result = float(m.group(1))
        except AttributeError:
            result = 0
        return result


if __name__ == '__main__':
    bleu = MtEval_BLEU(DefaultConfig())
    dummy_result = [['a','b','c'],['e', 'f', 'g']]
    x = bleu.get_bleu_from_decoderesult(dummy_result, dummy_result, dummy_result)
    print(x)

