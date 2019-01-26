import spacy
from collections import defaultdict
from datetime import datetime
from os.path import exists
from os import mkdir
from multiprocessing import Pool

nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger'])

doc = nlp('Jeddah is the principal gateway to Mecca , Islam \'s holiest city , which able-bodied Muslims are required to visit at least once in their lifetime .')
for word in doc:
    print(word)