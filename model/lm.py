class LM:
    def __init__(self):
        self.path_ngram = '/Users/zhaosanqiang916/Downloads/googlebooks-eng-all-5gram-20120701-en'

    def init_ngram(self):
        self.tree = {}
        for line in open(self.path_ngram, encoding='utf-8'):
            words = line.split('\t')[0].split(' ')
            words = [word[0] for word in words.split('_')]

            node = self.tree
            for i in range(len(words) - 1):
                word = words[i]
                if word not in node:
                    node[word] = {}
                node = node[word]
            # Last
            word = words[-1]
            if word not in node:
                node[word] = 0
            node[word] += 1
