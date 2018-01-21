if False:
    import tensorflow as tf
    sess = tf.Session()

    # indices = tf.constant([0], shape=(1,1))
    # data = tf.Variable([[1,2,3],[11,12,13],[11,12,13],[11,12,13],[11,12,13],[11,12,13]])
    # data_subset = tf.constant([111,112,113], shape=(1,3))
    # data = tf.scatter_nd_update(data, indices, data_subset)


    indices = tf.constant([[0]])
    data = tf.Variable([[1],[9],[19]])
    data = tf.scatter_nd_add(data, indices, tf.constant([[1]]))


    sess.run(tf.global_variables_initializer())
    print(sess.run(data))

if True:
    f = open('/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilarge2/src.rules.txt')
    rule2idx = {}
    idx2rule = []
    for line in f:
        rules = line.split('\t')
        for rule in rules:
            rule_items = rule.split('=>')
            if len(rule_items) == 4:
                rule = rule_items[1] + '=>' + rule_items[2]
                if rule not in rule2idx:
                    rule2idx[rule] = 0
                rule2idx[rule] += 1

    print(len(rule2idx))