import os

class DefaultConfig():
    use_gpu = True
    batch_size = 2
    dimension = 512
    max_complex_sentence = 15
    max_simple_sentence = 15
    model_save_freq = 100

    min_count = 0
    lower_case = True
    tokenizer = 'split' # ws: white space split / nltk: nltk tokenizer

    # Follow the configuration from https://github.com/XingxingZhang/dress
    optimizer = 'adagrad'
    learning_rate = 0.01
    max_grad_staleness = 0.0
    max_grad_norm = 4.0

    beam_search_size = -1
    train_with_hyp = False

    # Overwrite transformer config


    # deprecated: std of trunc norm init, used for initializing embedding / w
    # trunc_norm_init_std = 1e-4

    # tie_embedding configuration description
    # all:encoder/decoder/output; dec_out: decoder/output; enc_dec: encoder/decoder
    # allt:encoder/decoder+transform/output+transform;
    # dec_out: decoder/output+transform;
    # enc_dec: encoder/decoder+transform
    # none: no tied embedding
    tie_embedding = 'all'

    train_dataset_simple = '../data/train_dummy_simple_dataset'
    train_dataset_complex = '../data/train_dummy_complex_dataset'
    vocab_simple = '../data/dummy_simple_vocab'
    vocab_complex = '../data/dummy_complex_vocab'
    vocab_all = '../data/dummy_vocab'

    val_dataset_simple_folder = '../data/'
    val_dataset_simple_file = 'valid_dummy_simple_dataset'
    val_dataset_complex = '../data/valid_dummy_complex_dataset'
    num_refs = 0

    logdir = '../../tmp/log/'
    outdir = '../../tmp/output/'
    modeldir = '../../tmp/model/'

    allow_growth = True
    per_process_gpu_memory_fraction = 1.0


class DefaultTrainConfig(DefaultConfig):
    beam_search_size = 0
    train_with_hyp = False


class DefaultTestConfig(DefaultConfig):
    beam_search_size = 4
    # train_with_hyp = True


class WikiTurk(DefaultConfig):
    train_dataset_simple = '../../text_simplification_data/train/sentence-aligned.v2/simple.aligned'
    train_dataset_complex = '../../text_simplification_data/train/sentence-aligned.v2/normal.aligned'
    vocab_simple = '../../text_simplification_data/train/sentence-aligned.v2/simple.voc'
    vocab_complex = '../../text_simplification_data/train/sentence-aligned.v2/normal.voc'
    vocab_all = '../../text_simplification_data/train/sentence-aligned.v2/all.voc'

    val_dataset_simple_folder = '../../text_simplification_data/val/'
    val_dataset_simple_file = 'tune.8turkers.tok.simp'
    val_dataset_simple_references = 'tune.8turkers.tok.turk.'
    val_dataset_complex = '../../text_simplification_data/val/tune.8turkers.tok.norm'
    num_refs = 8


class WikiTurkTrainConfig(WikiTurk):
    beam_search_size = 0
    train_with_hyp = False


class WikiTurkTestConfig(WikiTurk):
    beam_search_size = 0
    use_gpu = False


class WikiDressLargeDefault(DefaultConfig):
    train_dataset_simple = '../../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.dst'
    train_dataset_complex = '../../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src'
    vocab_simple = '../../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.dst.vocab'
    vocab_complex = '../../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src.vocab'
    vocab_all = '../..//text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.vocab'

    num_refs = 0
    val_dataset_simple_folder = '../../text_simplification_data/train/dress/wikilarge/'
    val_dataset_simple_file = 'wiki.full.aner.valid.dst'
    val_dataset_complex = '../../text_simplification_data/train/dress/wikilarge/wiki.full.aner.valid.src'

    dimension = 256
    max_complex_sentence = 85
    max_simple_sentence = 85
    min_count = 4
    batch_size = 25


class WikiDressLargeTrainConfig(WikiDressLargeDefault):
    beam_search_size = 0
    train_with_hyp = False


class WikiDressLargeTestConfig(WikiDressLargeDefault):
    beam_search_size = 4
    use_gpu = False
    batch_size = 1
