class DefaultConfig():
    batch_size = 25
    dimension = 512
    max_complex_sentence = 100
    max_simple_sentence = 100

    optimizer = 'adagrad'
    learning_rate = 0.001
    max_grad_staleness = 0.0
    max_grad_norm = 4.0

    dataset_simple = '../data/dummy_simple_dataset'
    dataset_complex = '../data/dummy_complex_dataset'
    vocab_simple = '../data/dummy_simple_vocab'
    vocab_complex = '../data/dummy_complex_vocab'

    logdir = '../../tmp'
    outdir = '../../tmp'
