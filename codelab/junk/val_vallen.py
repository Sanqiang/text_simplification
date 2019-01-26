l = -1
for line in open('/Users/sanqiangzhao/git/ts/text_simplification_data/test/test.8turkers.tok.norm'):
    words = line.split()
    for word in words:
        if len(word) > 15:
            print(word)

print(len('http://www.genealogiafamiliar.net/'))