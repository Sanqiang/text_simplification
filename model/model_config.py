import os
from util.arguments import get_args


args = get_args()


def get_path(file_path):
    return os.path.dirname(os.path.abspath(__file__)) + '/../' + file_path
    # return os.getcwd() + '/' + file_path


class DefaultConfig():
    framework = args.framework
    warm_start = args.warm_start
    use_gpu = True
    batch_size = 3
    dimension = 32
    num_heads = 4
    max_complex_sentence = 20
    max_simple_sentence = 15
    min_simple_sentence = 5 #Used for Beam Search
    model_save_freq = 1000
    save_model_secs = 60

    min_count = 0
    lower_case = True
    tokenizer = 'split' # split: white space split / nltk: nltk tokenizer

    # Follow the configuration from https://github.com/XingxingZhang/dress
    optimizer = 'adagrad'
    learning_rate = 0.01
    max_grad_staleness = 0.0
    max_grad_norm = 5.0

    beam_search_size = -1
    train_with_hyp = False

    # Overwrite transformer config
    # timing: use positional encoding
    hparams_pos = args.hparams_pos

    # data quality model
    use_quality_model = args.use_quality_model

    # post process
    replace_unk_by_attn = False
    replace_unk_by_emb = True
    replace_unk_by_cnt = False
    replace_ner = True

    # deprecated: std of trunc norm init, used for initializing embedding / w
    # trunc_norm_init_std = 1e-4

    # tie_embedding configuration description
    # all:encoder/decoder/output; dec_out: decoder/output; enc_dec: encoder/decoder
    # non-implemented/allt:encoder/decoder+transform/output+transform;
    # non-implemented/dec_outt: decoder/output+transform;
    # non-implemented/enc_dect: encoder/decoder+transform
    # none: no tied embedding
    tie_embedding = args.tied_embedding
    pretrained_embedding = None

    train_dataset_simple = get_path('data/train_dummy_simple_dataset')
    train_dataset_complex = get_path('data/train_dummy_complex_dataset')
    vocab_simple = get_path('data/dummy_simple_vocab')
    vocab_complex = get_path('data/dummy_complex_vocab')
    vocab_all = get_path('data/dummy_vocab')

    val_dataset_simple_folder = get_path('data/')
    val_dataset_simple_file = 'valid_dummy_simple_dataset'
    val_dataset_complex = get_path('data/valid_dummy_complex_dataset')
    val_dataset_simple_references = 'valid_dummy_simple_dataset.'
    val_mapper = get_path('data/valid_dummy_mapper')
    num_refs = 3

    output_folder = args.output_folder
    logdir = get_path('../' + output_folder + '/log/')
    outdir = get_path('../' + output_folder + '/output/')
    modeldir = get_path('../' + output_folder + '/model/')

    allow_growth = False
    # per_process_gpu_memory_fraction = 1.0

    use_mteval = True
    mteval_script = get_path('script/mteval-v13a.pl')


class DefaultTrainConfig(DefaultConfig):
    beam_search_size = 0
    train_with_hyp = False


class DefaultTestConfig(DefaultConfig):
    beam_search_size = 4
    batch_size = 2
    train_with_hyp = False


class WikiTurk(DefaultConfig):
    train_dataset_simple = get_path('../text_simplification_data/train/sentence-aligned.v2/simple.aligned')
    train_dataset_complex = get_path('../text_simplification_data/train/sentence-aligned.v2/normal.aligned')
    vocab_simple = get_path('../text_simplification_data/train/sentence-aligned.v2/simple.voc')
    vocab_complex = get_path('../text_simplification_data/train/sentence-aligned.v2/normal.voc')
    vocab_all = get_path('../text_simplification_data/train/sentence-aligned.v2/all.voc')

    val_dataset_simple_folder = get_path('../text_simplification_data/val/')
    val_dataset_simple_file = 'tune.8turkers.tok.simp'
    val_dataset_simple_references = 'tune.8turkers.tok.turk.'
    val_dataset_complex = get_path('../text_simplification_data/val/tune.8turkers.tok.norm')
    num_refs = 8


class WikiTurkTrainConfig(WikiTurk):
    beam_search_size = 0
    train_with_hyp = False


class WikiTurkTestConfig(WikiTurk):
    beam_search_size = 0
    use_gpu = False


class WikiDressLargeDefault(DefaultConfig):
    train_dataset_simple = get_path('../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.dst')
    train_dataset_complex = get_path('../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src')
    vocab_simple = get_path('../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.dst.vocab')
    vocab_complex = get_path('../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src.vocab')
    vocab_all = get_path('../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.vocab')

    # num_refs = 0
    # val_dataset_simple_folder = get_path('../text_simplification_data/train/dress/wikilarge/')
    # val_dataset_simple_file = 'wiki.full.aner.valid.dst'
    # val_dataset_complex = get_path('../text_simplification_data/train/dress/wikilarge/wiki.full.aner.valid.src')
    val_dataset_simple_folder = get_path('../text_simplification_data/val/')
    val_dataset_simple_file = 'tune.8turkers.tok.simp.processed'
    val_dataset_simple_references = 'tune.8turkers.tok.turk.processed.'
    val_dataset_complex = get_path('../text_simplification_data/val/tune.8turkers.tok.norm.processed')
    val_mapper = get_path('../text_simplification_data/val/tune.8turkers.tok.map')
    val_dataset_simple_raw_file = 'tune.8turkers.tok.simp.raw'
    val_dataset_simple_raw_references = 'tune.8turkers.tok.turk.raw.'
    val_dataset_complex_raw = get_path('../text_simplification_data/val/tune.8turkers.tok.norm.raw')
    num_refs = 8

    save_model_secs = 600

    dimension = 300
    num_heads = 5
    max_complex_sentence = 85
    max_simple_sentence = 85
    min_count = 5
    batch_size = 86
    model_save_freq = 1000

    tokenizer = 'split'
    pretrained_embedding = get_path('../text_simplification_data/glove/glove.840B.300d.txt')
    pretrained_embedding_simple = get_path(
        '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.dst.vocab.pretrained')
    pretrained_embedding_complex = get_path(
        '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src.vocab.pretrained')


class WikiDressLargeTrainConfig(WikiDressLargeDefault):
    beam_search_size = 0
    train_with_hyp = False


class WikiDressLargeTestConfig(WikiDressLargeDefault):
    beam_search_size = 1
    batch_size = 86
    replace_unk_by_emb = True
    train_with_hyp = True


def list_config(config):
    attrs = [attr for attr in dir(config)
               if not callable(getattr(config, attr)) and not attr.startswith("__")]
    output = ''
    for attr in attrs:
        val = getattr(config, attr)
        output = '\n'.join([output, '%s=\t%s' % (attr, val)])
    return output
