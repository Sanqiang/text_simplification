import time

def gen():
    i = 0
    while True:
        yield i
        i += 1


it = gen()
while True:
    x = next(it)
    print(x)
    time.sleep(1)
