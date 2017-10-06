line_sep = '\n'
sample_sep = '\t'
kv_sep = '=>'


def dump_mappers(mappers, path):
    output = ''
    for mapper in mappers:
        tmp = ''
        for k in mapper:
            v = mapper[k]
            tmp = sample_sep.join([tmp, str(k) + kv_sep + str(v)])
        output = line_sep.join([output, tmp])

    #Remove first white space
    output = output[len(line_sep):]

    f = open(path, 'w', encoding='utf-8')
    f.write(output)
    f.close()


def load_mappers(path, lower_case=False):
    mappers = []
    f = open(path, encoding='utf-8')
    for line in f:
        tmp_mapper = {}
        samples = line.strip().split('\t')
        for sample in samples:
            kv = sample.split(kv_sep)
            if len(kv) == 2:
                v = kv[0]
                k = kv[1]
                if lower_case:
                    v = v.lower()
                    k = k.lower()
                tmp_mapper[k] = v
        mappers.append(tmp_mapper)
    return mappers

