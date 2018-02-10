import math
from nltk.tokenize import RegexpTokenizer

from util.mteval_bleu import MtEval_BLEU

class Readability:
    analyzedVars = {}

    def __init__(self, text):
        self.analyze_text(text)

    def analyze_text(self, text):
        words = get_words(text)
        char_count = get_char_count(words)
        word_count = len(words)
        sentence_count = len(get_sentences(text))
        syllable_count = count_syllables(words)
        complexwords_count = count_complex_words(text)
        avg_words_p_sentence = word_count / sentence_count

        self.analyzedVars = {
            'words': words,
            'char_cnt': float(char_count),
            'word_cnt': float(word_count),
            'sentence_cnt': float(sentence_count),
            'syllable_cnt': float(syllable_count),
            'complex_word_cnt': float(complexwords_count),
            'avg_words_p_sentence': float(avg_words_p_sentence)
        }

    def ARI(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = 4.71 * (self.analyzedVars['char_cnt'] / self.analyzedVars['word_cnt']) + 0.5 * (
            self.analyzedVars['word_cnt'] / self.analyzedVars['sentence_cnt']) - 21.43
        return score

    def FleschReadingEase(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = 206.835 - (1.015 * (self.analyzedVars['avg_words_p_sentence'])) - (
            84.6 * (self.analyzedVars['syllable_cnt'] / self.analyzedVars['word_cnt']))
        return round(score, 4)

    def FleschKincaidGradeLevel(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = 0.39 * (self.analyzedVars['avg_words_p_sentence']) + 11.8 * (
            self.analyzedVars['syllable_cnt'] / self.analyzedVars['word_cnt']) - 15.59
        return round(score, 4)

    def GunningFogIndex(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = 0.4 * ((self.analyzedVars['avg_words_p_sentence']) + (
            100 * (self.analyzedVars['complex_word_cnt'] / self.analyzedVars['word_cnt'])))
        return round(score, 4)

    def SMOGIndex(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = (math.sqrt(self.analyzedVars['complex_word_cnt'] * (30 / self.analyzedVars['sentence_cnt'])) + 3)
        return score

    def ColemanLiauIndex(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = (5.89 * (self.analyzedVars['char_cnt'] / self.analyzedVars['word_cnt'])) - (
            30 * (self.analyzedVars['sentence_cnt'] / self.analyzedVars['word_cnt'])) - 15.8
        return round(score, 4)

    def LIX(self):
        longwords = 0.0
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            for word in self.analyzedVars['words']:
                if len(word) >= 7:
                    longwords += 1.0
            score = self.analyzedVars['word_cnt'] / self.analyzedVars['sentence_cnt'] + float(100 * longwords) / \
                                                                                        self.analyzedVars['word_cnt']
        return score

    def RIX(self):
        longwords = 0.0
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            for word in self.analyzedVars['words']:
                if len(word) >= 7:
                    longwords += 1.0
            score = longwords / self.analyzedVars['sentence_cnt']
        return score


import nltk

TOKENIZER = RegexpTokenizer('(?u)\W+|\$[\d\.]+|\S+')
SPECIAL_CHARS = ['.', ',', '!', '?']


def get_char_count(words):
    characters = 0
    for word in words:
        characters += len(word)
    return characters


def get_words(text=''):
    words = []
    words = TOKENIZER.tokenize(text)
    filtered_words = []
    for word in words:
        if word in SPECIAL_CHARS or word == " ":
            pass
        else:
            new_word = word.replace(",", "").replace(".", "")
            new_word = new_word.replace("!", "").replace("?", "")
            filtered_words.append(new_word)
    return filtered_words


def get_sentences(text=''):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    # sentences = tokenizer.tokenize(text)
    sentences = tokenizer.tokenize(text)
    return sentences


def count_syllables(words):
    syllableCount = 0
    for word in words:
        syllableCount += count(word)
    return syllableCount


# This method must be enhanced. At the moment it only
# considers the number of syllables in a word.
# This often results in that too many complex words are detected.
def count_complex_words(text=''):
    words = get_words(text)
    sentences = get_sentences(text)
    complex_words = 0
    found = False
    cur_word = []

    for word in words:
        cur_word.append(word)
        if count_syllables(cur_word) >= 3:

            # Checking proper nouns. If a word starts with a capital letter
            # and is NOT at the beginning of a sentence we don't add it
            # as a complex word.
            if not (word[0].isupper()):
                complex_words += 1
            else:
                for sentence in sentences:
                    # if str(sentence).startswith(word):
                    if sentence.startswith(word):
                        found = True
                        break
                if found:
                    complex_words += 1
                    found = False

        cur_word.remove(word)
    return complex_words

"""
Fallback syllable counter
This is based on the algorithm in Greg Fast's perl module
Lingua::EN::Syllable.
"""

import string, re, os

specialSyllables_en = """tottered 2
chummed 1
peeped 1
moustaches 2
shamefully 3
messieurs 2
satiated 4
sailmaker 4
sheered 1
disinterred 3
propitiatory 6
bepatched 2
particularized 5
caressed 2
trespassed 2
sepulchre 3
flapped 1
hemispheres 3
pencilled 2
motioned 2
poleman 2
slandered 2
sombre 2
etc 4
sidespring 2
mimes 1
effaces 2
mr 2
mrs 2
ms 1
dr 2
st 1
sr 2
jr 2
truckle 2
foamed 1
fringed 2
clattered 2
capered 2
mangroves 2
suavely 2
reclined 2
brutes 1
effaced 2
quivered 2
h'm 1
veriest 3
sententiously 4
deafened 2
manoeuvred 3
unstained 2
gaped 1
stammered 2
shivered 2
discoloured 3
gravesend 2
60 2
lb 1
unexpressed 3
greyish 2
unostentatious 5
"""

fallback_cache = {}

fallback_subsyl = ["cial", "tia", "cius", "cious", "gui", "ion", "iou",
                   "sia$", ".ely$"]

fallback_addsyl = ["ia", "riet", "dien", "iu", "io", "ii",
                   "[aeiouy]bl$", "mbl$",
                   "[aeiou]{3}",
                   "^mc", "ism$",
                   "(.)(?!\\1)([aeiouy])\\2l$",
                   "[^l]llien",
                   "^coad.", "^coag.", "^coal.", "^coax.",
                   "(.)(?!\\1)[gq]ua(.)(?!\\2)[aeiou]",
                   "dnt$"]


# Compile our regular expressions
for i in range(len(fallback_subsyl)):
    fallback_subsyl[i] = re.compile(fallback_subsyl[i])
for i in range(len(fallback_addsyl)):
    fallback_addsyl[i] = re.compile(fallback_addsyl[i])

def _normalize_word(word):
    return word.strip().lower()

# Read our syllable override file and stash that info in the cache
for line in specialSyllables_en.splitlines():
    line = line.strip()
    if line:
        toks = line.split()
        assert len(toks) == 2
        fallback_cache[_normalize_word(toks[0])] = int(toks[1])

def count(word):
    word = _normalize_word(word)
    if not word:
        return 0

    # Check for a cached syllable count
    count = fallback_cache.get(word, -1)
    if count > 0:
        return count

    # Remove final silent 'e'
    if word[-1] == "e":
        word = word[:-1]

    # Count vowel groups
    count = 0
    prev_was_vowel = 0
    for c in word:
        is_vowel = c in ("a", "e", "i", "o", "u", "y")
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel

    # Add & subtract syllables
    for r in fallback_addsyl:
        if r.search(word):
            count += 1
    for r in fallback_subsyl:
        if r.search(word):
            count -= 1

    # Cache the syllable count
    fallback_cache[word] = count

    return count


def get_fkgl(text):
    try:
        rd = Readability(text)
        fkgl = rd.FleschKincaidGradeLevel()
    except:
        # Large penalty
        fkgl = 100
    return fkgl


class CorpusFKGL(MtEval_BLEU):

    def get_fkgl_from_joshua(self, step, targets):
        path_tar = self.model_config.resultdir + '/joshua_target_%s.txt' % step
        if not os.path.exists(path_tar):
            f = open(path_tar, 'w', encoding='utf-8')
            # joshua require lower case
            f.write(self.result2txt(targets, lowercase=True))
            f.close()

        text = open(path_tar, encoding='utf-8').read()
        return get_fkgl(text)



if __name__ == "__main__":
    text = """We are close to wrapping up our 10 week Rails Course. This week we will cover a handful of topics commonly encountered in Rails projects. We then wrap up with part 2 of our Reddit on Rails exercise!  By now you should be hard at work on your personal projects. The students in the course just presented in front of the class with some live demos and a brief intro to to the problems their app were solving. Maybe set aside some time this week to show someone your progress, block off 5 minutes and describe what goal you are working towards, the current state of the project (is it almost done, just getting started, needs UI, etc.), and then show them a quick demo of the app. Explain what type of feedback you are looking for (conceptual, design, usability, etc.) and see what they have to say.  As we are wrapping up the course you need to be focused on learning as much as you can, but also making sure you have the tools to succeed after the class is over."""

    rd = Readability(text)
    print('Test text:')
    print('"%s"\n' % text)
    print('ARI: ', rd.ARI())
    print('FleschReadingEase: ', rd.FleschReadingEase())
    print('FleschKincaidGradeLevel: ', rd.FleschKincaidGradeLevel())
    print('GunningFogIndex: ', rd.GunningFogIndex())
    print('SMOGIndex: ', rd.SMOGIndex())
    print('ColemanLiauIndex: ', rd.ColemanLiauIndex())
    print('LIX: ', rd.LIX())
    print('RIX: ', rd.RIX())

