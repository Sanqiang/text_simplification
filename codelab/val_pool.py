from multiprocessing.dummy import Pool as ThreadPool

l1 = [1,2,3,4,5]
l2 = [6,7,8,9,0]

def p(i):
    i1, i2 = i
    return i1+i2

pool = ThreadPool(4)
r = pool.map(p, zip(l1, l2))
print(r)