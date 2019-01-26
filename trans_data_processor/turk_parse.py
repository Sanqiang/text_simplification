"""Process Val/Test data for turk."""
import spacy

RAW_PATH = '/Users/sanqiangzhao/git/ts/text_simplification_data/test_0930/test.8turkers.organized.tsv'
NEW_PATH = '/Users/sanqiangzhao/git/ts/text_simplification_data/test_0930/'
COMP_PREFIX = 'lower.norm.ori'
SIMP_PREFIX = 'lower.ref.ori.'

nlp = spacy.load('en_core_web_lg', disable=['parser'])
def _process_sent(sent):
    doc = nlp(sent)
    return ' '.join([w.text for w in doc]).strip().lower()

(comp_sents, ref1_sents, ref2_sents, ref3_sents,
 ref4_sents, ref5_sents, ref6_sents, ref7_sents, ref8_sents) = (
    [], [], [], [], [], [], [], [], [])

for line in open(RAW_PATH):
    items = line.split('\t')

    comp_sent = items[1]
    (ref1_sent, ref2_sent, ref3_sent, ref4_sent,
     ref5_sent, ref6_sent, ref7_sent,ref8_sent) = (
        items[2].strip(), items[3].strip(), items[4].strip(), items[5].strip(),
        items[6].strip(), items[7].strip(), items[8].strip(), items[9].strip())

    comp_sents.append(_process_sent(comp_sent))
    ref1_sents.append(_process_sent(ref1_sent))
    ref2_sents.append(_process_sent(ref2_sent))
    ref3_sents.append(_process_sent(ref3_sent))
    ref4_sents.append(_process_sent(ref4_sent))
    ref5_sents.append(_process_sent(ref5_sent))
    ref6_sents.append(_process_sent(ref6_sent))
    ref7_sents.append(_process_sent(ref7_sent))
    ref8_sents.append(_process_sent(ref8_sent))


open(NEW_PATH + COMP_PREFIX, 'w').write('\n'.join(comp_sents))
open(NEW_PATH + SIMP_PREFIX + '0', 'w').write('\n'.join(ref1_sents))
open(NEW_PATH + SIMP_PREFIX + '1', 'w').write('\n'.join(ref2_sents))
open(NEW_PATH + SIMP_PREFIX + '2', 'w').write('\n'.join(ref3_sents))
open(NEW_PATH + SIMP_PREFIX + '3', 'w').write('\n'.join(ref4_sents))
open(NEW_PATH + SIMP_PREFIX + '4', 'w').write('\n'.join(ref5_sents))
open(NEW_PATH + SIMP_PREFIX + '5', 'w').write('\n'.join(ref6_sents))
open(NEW_PATH + SIMP_PREFIX + '6', 'w').write('\n'.join(ref7_sents))
open(NEW_PATH + SIMP_PREFIX + '7', 'w').write('\n'.join(ref8_sents))