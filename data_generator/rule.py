

class Rule:
    def __init__(self, model_config, rule_path=None):
        self.model_config = model_config
        self.rule_path = rule_path
        self.populate_rulebase(model_config.rule_threshold)

    def populate_rulebase(self, minscore):
        self.r2i = {'pad': 0}
        self.i2r = ['pad']
        self.r2freq = {}

        for line in open(self.rule_path, encoding='utf-8'):
            items = line.strip().split('\t')
            w = items[0]

            # if self.model_config.rule_mode == 'unigram':
            #     pair = w.split('=>')
            #     ori = pair[0].split()
            #     tar = pair[1].split()
            #     if len(ori) > 1 or len(tar) > 1:
            #         continue

            score = 0
            if len(items) > 1:
                score = float(items[1])
            if score >= minscore:
                self.r2i[w] = len(self.i2r)
                self.i2r.append(w)
            self.r2freq[w] = score

        print('Rule Populated with size %d for path %s.'
              % (len(self.i2r), self.rule_path))

    def get_rule_size(self):
        return len(self.i2r)

    def encode(self, rule):
        if not rule:
            return None, None, None
        rule_items = rule.split('=>')
        rule_pair = rule_items[0] + '=>' + rule_items[1]
        if rule_pair in self.r2i:
            return self.r2i[rule_pair], rule_items[0], rule_items[1]
        return None, None, None

    def contain(self, rule):
        rule_items = rule.split('=>')
        rule_pair = rule_items[0] + '=>' + rule_items[1]
        if rule_pair in self.r2i:
            return True
        return False

    def get_freq(self, rule):
        if not rule:
            return 0
        rule_items = rule.split('=>')
        rule_pair = rule_items[0] + '=>' + rule_items[1]
        if rule_pair in self.r2freq:
            return self.r2freq[rule_pair]
        return 0
