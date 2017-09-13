class DefaultConfig():
    batch_size = 2
    dimension = 512
    max_complex_sentence = 50
    max_simple_sentence = 50

    optimizer = 'adagrad'
    learning_rate = 0.001
    max_grad_staleness = 0.0
    max_grad_norm = 4.0
    beam_search_size = 0

    dataset_simple = '../data/dummy_simple_dataset'
    dataset_complex = '../data/dummy_complex_dataset'
    vocab_simple = '../data/dummy_simple_vocab'
    vocab_complex = '../data/dummy_complex_vocab'

    logdir = '../../tmp'
    outdir = '../../tmp'
