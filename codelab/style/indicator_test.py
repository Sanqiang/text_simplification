import spacy


nlp = spacy.load('en_core_web_lg', disable=['tagger', 'ner'])

sent = 'after the jerilderie raid , the gang laid low for 16 months evacapture .'
doc = nlp(sent)

for w in doc:
    print(w.dep_)