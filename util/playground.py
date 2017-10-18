from util.sari import SARIsent


if __name__ == '__main__':
    ssent = 'my name is sanqiang .'
    cent1 = 'i'
    cent2 = 'sanqiang'
    rsent = ['i am sanqiang .']

    v1 = SARIsent(ssent, cent1, rsent)
    v2 = SARIsent(ssent, cent2, rsent)
    print('v1:\t%s' % v1)
    print('v2:\t%s' % v2)