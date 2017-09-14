class DefaultConfig():
    batch_size = 3
    dimension = 32
    max_complex_sentence = 10
    max_simple_sentence = 10
    model_save_freq = 3000

    optimizer = 'adagrad'
    learning_rate = 0.001
    max_grad_staleness = 0.0
    max_grad_norm = 4.0
    beam_search_size = -1
    train_with_hyp = False

    dataset_simple = '../data/dummy_simple_dataset'
    dataset_complex = '../data/dummy_complex_dataset'
    vocab_simple = '../data/dummy_simple_vocab'
    vocab_complex = '../data/dummy_complex_vocab'

    logdir = '../../tmp'
    outdir = '../../tmp'

class DefaultTrainConfig(DefaultConfig):
    beam_search_size = 1
    train_with_hyp = True

class DefaultTestConfig(DefaultConfig):
    beam_search_size = 1

