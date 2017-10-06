import os
from util.arguments import get_args


args = get_args()


def get_path(file_path):
    return os.path.dirname(os.path.abspath(__file__)) + '/../' + file_path


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
    optimizer = args.optimizer
    learning_rate_warmup_steps = 50000
    use_learning_rate_decay = True
    learning_rate = args.learning_rate
    max_grad_staleness = 0.0
    max_grad_norm = 2.0
    loss_fn = args.loss_fn
    num_samples = args.number_samples

    beam_search_size = -1

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
    val_dataset_simple_raw_file = val_dataset_simple_file
    val_dataset_simple_raw_references = val_dataset_simple_references
    val_dataset_complex_raw = val_dataset_complex
    num_refs = 3

    output_folder = args.output_folder
    logdir = get_path('../' + output_folder + '/log/')
    outdir = get_path('../' + output_folder + '/output/')
    modeldir = get_path('../' + output_folder + '/model/')
    resultdor = get_path('../' + output_folder + '/result/')

    allow_growth = False
    # per_process_gpu_memory_fraction = 1.0

    use_mteval = True
    mteval_script = get_path('script/mteval-v13a.pl')


class DefaultTrainConfig(DefaultConfig):
    beam_search_size = 0


class DefaultTestConfig(DefaultConfig):
    beam_search_size = 4
    batch_size = 2
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/test1')


class DefaultTestConfig2(DefaultConfig):
    beam_search_size = 4
    batch_size = 1
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/test2')


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
    min_count = args.min_count
    batch_size = 32
    model_save_freq = 1000

    tokenizer = 'split'
    pretrained_embedding = get_path('../text_simplification_data/glove/glove.840B.300d.txt')
    pretrained_embedding_simple = get_path(
        '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.dst.vocab.pretrained')
    pretrained_embedding_complex = get_path(
        '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src.vocab.pretrained')


class WikiDressLargeTrainConfig(WikiDressLargeDefault):
    beam_search_size = 0


class WikiDressLargeTestConfig(WikiDressLargeDefault):
    beam_search_size = 1
    batch_size = 64
    replace_unk_by_emb = True


class SubTest(WikiDressLargeDefault):
    batch_size = 64
    replace_unk_by_emb = True
    beam_search_size = 1
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/')


class SubValWikiEightRefConfig(SubTest):
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/eightref_val')

    val_dataset_simple_folder = get_path('../text_simplification_data/val/')
    val_dataset_simple_file = 'tune.8turkers.tok.simp.processed'
    val_dataset_simple_references = 'tune.8turkers.tok.turk.processed.'
    val_dataset_complex = get_path('../text_simplification_data/val/tune.8turkers.tok.norm.processed')
    val_mapper = get_path('../text_simplification_data/val/tune.8turkers.tok.map')
    val_dataset_simple_raw_file = 'tune.8turkers.tok.simp.raw'
    val_dataset_simple_raw_references = 'tune.8turkers.tok.turk.raw.'
    val_dataset_complex_raw = get_path('../text_simplification_data/val/tune.8turkers.tok.norm.raw')
    num_refs = 8

class SubValWikiEightRefConfigBeam2(SubValWikiEightRefConfig):
    beam_search_size = 2
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/eightref_val_2head')


class SubTestWikiEightRefConfig(SubTest):
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/eightref_test')

    val_dataset_simple_folder = get_path('../text_simplification_data/test/')
    val_dataset_simple_file = 'test.8turkers.tok.simp.processed'
    val_dataset_simple_references = 'test.8turkers.tok.turk.processed.'
    val_dataset_complex = get_path('../text_simplification_data/test/test.8turkers.tok.norm.processed')
    val_mapper = get_path('../text_simplification_data/test/test.8turkers.tok.map')
    val_dataset_simple_raw_file = 'test.8turkers.tok.simp.raw'
    val_dataset_simple_raw_references = 'test.8turkers.tok.turk.raw.'
    val_dataset_complex_raw = get_path('../text_simplification_data/test/test.8turkers.tok.norm.raw')
    num_refs = 8


class SubTestWikiEightRefConfigBeam2(SubTestWikiEightRefConfig):
    beam_search_size = 2
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/eightref_test_2head')


class SubValWikiDress(SubTest):
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/dresssmall_val')

    val_dataset_simple_folder = get_path(
        '../text_simplification_data/train/dress/wikismall/')
    val_dataset_simple_file = 'PWKP_108016.tag.80.aner.ori.valid.dst.processed'
    val_dataset_complex = get_path(
        '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.ori.valid.src.processed')
    val_mapper = get_path(
        '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.ori.valid.map')
    val_dataset_simple_raw_file = 'PWKP_108016.tag.80.aner.ori.valid.dst'
    val_dataset_complex_raw = get_path(
        '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.ori.valid.src')
    num_refs = 0


class SubTestWikiDress(SubTest):
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/dresssmall_test')

    val_dataset_simple_folder = get_path(
        '../text_simplification_data/train/dress/wikismall/')
    val_dataset_simple_file = 'PWKP_108016.tag.80.aner.ori.test.dst.processed'
    val_dataset_complex = get_path(
        '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.ori.test.src.processed')
    val_mapper = get_path(
        '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.ori.test.map')
    val_dataset_simple_raw_file = 'PWKP_108016.tag.80.aner.ori.test.dst'
    val_dataset_complex_raw = get_path(
        '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.ori.test.src')
    num_refs = 0


class SubValWikiDressL(SubTest):
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/dresslarge_val')

    val_dataset_simple_folder = get_path(
        '../text_simplification_data/train/dress/wikilarge/')
    val_dataset_simple_file = 'wiki.full.aner.valid.dst.processed'
    val_dataset_complex = get_path(
        '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.valid.src.processed')
    val_mapper = get_path(
        '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.valid.map')
    val_dataset_simple_raw_file = 'wiki.full.aner.valid.dst'
    val_dataset_complex_raw = get_path(
        '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.valid.src')
    num_refs = 0


class SubTestWikiDressL(SubTest):
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/dresslarge_test')

    val_dataset_simple_folder = get_path(
        '../text_simplification_data/train/dress/wikilarge/')
    val_dataset_simple_file = 'wiki.full.aner.test.dst.processed'
    val_dataset_complex = get_path(
        '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.test.src.processed')
    val_mapper = get_path(
        '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.test.map')
    val_dataset_simple_raw_file = 'wiki.full.aner.test.dst'
    val_dataset_complex_raw = get_path(
        '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.test.src')
    num_refs = 0


class SubValWikiDressBeam2(SubValWikiDress):
    beam_search_size = 2
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/dresssmall_val_2head')


class SubTestWikiDressBeam2(SubTestWikiDress):
    beam_search_size = 2
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/dresssmall_test_2head')


class SubValWikiDressLBeam2(SubValWikiDressL):
    beam_search_size = 2
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/dresslarge_val_2head')

class SubTestWikiDressLBeam2(SubTestWikiDressL):
    beam_search_size = 2
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/dresslarge_test_2head')


def list_config(config):
    attrs = [attr for attr in dir(config)
               if not callable(getattr(config, attr)) and not attr.startswith("__")]
    output = ''
    for attr in attrs:
        val = getattr(config, attr)
        output = '\n'.join([output, '%s=\t%s' % (attr, val)])
    return output
