import json
import os


PATH = '/Users/sanqiang/git/ts/text_simplification_data/sentcompress/'
TAG = 'train'
OUTPUT_PATH = os.path.join('/Users/sanqiang/git/ts/text_simplification_data/sentcompress/', TAG)

sent_comps, sent_simps = [], []
for file in os.listdir(os.path.join(PATH, 'download')):
    if not file.endswith('.json') or TAG not in file:
        continue

    buffer = []
    for line in open(os.path.join(PATH, 'download', file)):
        if line.strip():
            buffer.append(line)
        else:
            try:
                obj = json.loads(''.join(buffer))
                sent_comp = obj['graph']['sentence']
                sent_simp = obj['compression']['text']
                sent_comps.append(sent_comp)
                sent_simps.append(sent_simp)
            except:
                print(''.join(buffer))
                pass
            buffer.clear()

open(os.path.join(OUTPUT_PATH, 'ori.%s.src'%TAG), 'w').write('\n'.join(sent_comps))
open(os.path.join(OUTPUT_PATH, 'ori.%s.dst'%TAG), 'w').write('\n'.join(sent_simps))

