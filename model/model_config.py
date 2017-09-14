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

    train_dataset_simple = '../data/dummy_simple_dataset'
    train_dataset_complex = '../data/dummy_complex_dataset'
    vocab_simple = '../data/dummy_simple_vocab'
    vocab_complex = '../data/dummy_complex_vocab'

    val_dataset_simple_folder = '../../text_simplification_data/val/'
    val_dataset_simple_file = 'tune.8turkers.tok.simp'
    val_dataset_simple_references = 'tune.8turkers.tok.turk.'
    val_dataset_complex = '../../text_simplification_data/val/tune.8turkers.tok.norm'
    num_refs = 8

    logdir = '../../tmp/'
    outdir = '../../tmp/'
    modeldir = '../../tmp/model/'

class DefaultTrainConfig(DefaultConfig):
    beam_search_size = 1
    train_with_hyp = True

class DefaultTestConfig(DefaultConfig):
    beam_search_size = 3

