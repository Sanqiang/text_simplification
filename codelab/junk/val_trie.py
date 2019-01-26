import pygtrie

t = pygtrie.StringTrie(separator=' ')
t['xxx xx'] = [1,2,3]
t['xxx xx'].append(4)
# t['xxx a'].append(8)
t['xxx a'] = 3
t['xxxx'] = 2

print('xxx a' in t)
print(t.items(prefix='xxx'))