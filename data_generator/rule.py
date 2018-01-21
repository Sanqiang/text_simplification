

class Rule:
    def __init__(self, model_config, rule_path=None):
        self.model_config = model_config
        self.rule_path = rule_path
        self.populate_rulebase(0)

    def populate_rulebase(self, minscore):
        self.r2i = {'pad':0}
        self.i2r = ['pad']

        minscore = 0 #max(minscore, self.model_config.min_count)
        for line in open(self.rule_path, encoding='utf-8'):
            items = line.strip().split('\t')
            w = items[0]
            score = 0
            if len(items) > 1:
                score = float(items[1])
            if score >= minscore:
                self.r2i[w] = len(self.i2r)
                self.i2r.append(w)

        print('Rule Populated with size %d for path %s.'
              % (len(self.i2r), self.rule_path))

    def encode(self, rule):
        rule_items = rule.split('=>')
        if len(rule_items) == 4:
            rule_pair = rule_items[1] + '=>' + rule_items[2]
            if rule_pair in self.r2i:
                return self.r2i[rule_pair], rule_items[2].split()
        return None, None