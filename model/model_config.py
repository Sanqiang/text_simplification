import os
from util.arguments import get_args


args = get_args()


def get_path(file_path, zfs=False):
    if zfs:
        return "/zfs1/hdaqing/saz31/text_simplification/tmp/" + file_path
    else:
        return os.path.dirname(os.path.abspath(__file__)) + '/../' + file_path


class DefaultConfig():
    train_mode = args.train_mode
    num_gpus = args.num_gpus
    framework = args.framework
    warm_start = args.warm_start
    use_partial_restore = args.use_partial_restore
    use_gpu = True
    batch_size = 9
    dimension = 150
    max_complex_sentence = 10
    max_simple_sentence = 8
    # min_simple_sentence = 5 #Used for Beam Search
    model_eval_freq = args.model_eval_freq
    it_train = args.it_train
    model_print_freq = 1
    save_model_secs = 60
    number_samples = args.number_samples

    min_count = 0
    lower_case = args.lower_case
    tokenizer = 'split' # split: white space split / nltk: nltk tokenizer

    # Follow the configuration from https://github.com/XingxingZhang/dress
    optimizer = args.optimizer
    change_optimizer = args.change_optimizer
    learning_rate_warmup_steps = 50000
    learning_rate = args.learning_rate
    max_grad_norm = 4.0
    layer_prepostprocess_dropout = args.layer_prepostprocess_dropout

    # loss_fn = args.loss_fn
    # num_samples = args.number_samples

    beam_search_size = -1

    # Overwrite transformer config
    # timing: use positional encoding
    hparams_pos = args.hparams_pos

    num_heads = args.num_heads
    num_hidden_layers = args.num_hidden_layers
    num_encoder_layers = args.num_encoder_layers
    num_decoder_layers = args.num_decoder_layers

    # Our research ideas !!!
    # ppdb_mode: incorporate ppdb in our model
    # comp: use comp supervision, simp: use further simplify for simple sentence
    # none: no ppdb
    # ppdb_args: arguments for ppdb
    # for comp mode, empty uses rule weight
    # or 2.0|1.5 indicates 2.0 weight for SARI simple words and 1.5 weight for SARI keep words
    # or 2.0|1.5|1.0 the last 1.0 indicate enc_dec attention loss
    ppdb_mode = args.ppdb_mode
    ppdb_args = args.ppdb_args
    if ppdb_mode == 'comp' and ppdb_args is not None:
        ppdb_args = [float(w) for w in ppdb_args.split('|')]
    # ppdb_emode: incorporate ppdb in our model eval
    # none: no ppdb eval
    # weight: augment weight in logit
    # for weight ppdb eval mode args defines the weight to augment
    ppdb_emode = args.ppdb_emode
    ppdb_emode_args = args.ppdb_emode_args
    if ppdb_emode == 'weight':
        ppdb_emode_args = float(ppdb_emode_args)

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
    train_dataset_simple_ppdb = get_path('data/train_dummy_simple_dataset.rules')
    train_dataset_simple_syntax = get_path('data/train_dummy_simple_dataset.syntax')
    train_dataset_complex = get_path('data/train_dummy_complex_dataset')
    train_dataset_complex_ppdb = get_path('data/train_dummy_complex_dataset.rules')
    val_dataset_complex_ppdb = get_path('data/eval_dummy_complex_dataset.rules')
    vocab_simple = get_path('data/dummy_simple_vocab')
    vocab_complex = get_path('data/dummy_complex_vocab')
    vocab_all = get_path('data/dummy_vocab')
    vocab_rules = get_path('data/dummy_rules_vocab')
    if args.lower_case:
        vocab_simple = vocab_simple + '.lower'
        vocab_complex = vocab_complex + '.lower'
        vocab_all = vocab_all + '.lower'

    subword_vocab_size = args.subword_vocab_size
    subword_vocab_simple = vocab_simple + str(subword_vocab_size)
    subword_vocab_complex = vocab_complex + str(subword_vocab_size)
    subword_vocab_all = vocab_all + str(subword_vocab_size)

    if subword_vocab_size > 0:
        max_complex_sentence = 100
        max_simple_sentence = 90

    val_dataset_simple_folder = get_path('data/')
    val_dataset_simple_file = 'valid_dummy_simple_dataset'
    val_dataset_complex = get_path('data/valid_dummy_complex_dataset')
    val_mapper = get_path('data/valid_dummy_mapper')
    val_dataset_complex_rawlines_file = val_dataset_complex
    val_dataset_simple_rawlines_file_references = 'valid_dummy_simple_dataset.raw.'
    val_dataset_simple_rawlines_file = val_dataset_simple_file
    num_refs = 3

    output_folder = args.output_folder
    logdir = get_path('../' + output_folder + '/log/', True)
    modeldir = get_path('../' + output_folder + '/model/', True)
    resultdir = get_path('../' + output_folder + '/result/', True)

    allow_growth = True
    # per_process_gpu_memory_fraction = 1.0
    use_mteval = True
    mteval_script = get_path('script/mteval-v13a.pl')
    mteval_mul_script = get_path('script/multi-bleu.perl')
    joshua_class = get_path('script/ppdb-simplification-release-joshua5.0/joshua/class')
    joshua_script = get_path('script/ppdb-simplification-release-joshua5.0/joshua/bin/bleu')
    corpus_sari_script = get_path('script/corpus_sari.sh')
    corpus_sari_script_nonref = get_path('script/corpus_sari_nonref.sh')

    path_ppdb_refine = get_path(args.path_ppdb_refine)

    # For Exp
    penalty_alpha = args.penalty_alpha

    # For Memory
    memory = args.memory
    max_cand_rules = 15
    rule_size = 5
    memory_prepare_step = args.memory_prepare_step
    memory_config = args.memory_config
    min_count_rule = 0
    if 'mincnt' in memory_config:
        # Assume single digit for min_count_rule
        cnt_idx = memory_config.index('mincnt') + len("mincnt")
        min_count_rule = int(memory_config[cnt_idx: cnt_idx+1])
    ctxly = None
    if 'ctxly' in memory_config:
        ctxly_idx = memory_config.index('ctxly') + len("ctxly")
        ctxly = int(memory_config[ctxly_idx: ctxly_idx+1])

    # For RL
    rl_config = args.rl_config
    # sari: add sari metric for optimize
    # sari_ppdb_simp_weight: the weight for ppdb in sari
    # sample: sari|rule|rule_weight:2.0
    rl_configs = {}
    for cfg in rl_config.split('|'):
        kv = cfg.split(':')
        if kv[0] == 'sari':
            rl_configs['sari'] = True
        if kv[0] == 'sari_weight':
            rl_configs['sari_weight'] = float(kv[1])
        else:
            rl_configs['sari_weight'] = 1.0
        if kv[0] == 'rule':
            rl_configs['rule'] = True
        if kv[0] == 'rule_weight':
            rl_configs['rule_weight'] = float(kv[1])

    # RNN decoder
    rnn_decoder = args.rnn_decoder

    rule_base = args.rule_base

class DefaultTrainConfig(DefaultConfig):
    beam_search_size = 0


class DefaultTestConfig(DefaultConfig):
    beam_search_size = 1
    batch_size = 2
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/test1', True)


class DefaultTestConfig2(DefaultConfig):
    beam_search_size = 1
    batch_size = 2
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/test2')


class WikiDressLargeDefault(DefaultConfig):
    model_print_freq = 50
    save_model_secs = 600
    model_eval_freq = args.model_eval_freq

    train_dataset_simple = get_path('../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.dst')
    # train_dataset_simple_ppdb = get_path('../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.dst.rules')
    train_dataset_simple_syntax = get_path(
        '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.dst.jsyntax')
    train_dataset_complex_syntax = get_path(
        '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src.jsyntax')
    train_dataset_complex = get_path('../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src')
    val_dataset_complex_syntax = get_path(
        '../text_simplification_data/test/wiki.full.aner.test.src.jsyntax')
    rule_base = args.rule_base
    if rule_base == 'sing':
        # Only consider single word rules from v1
        train_dataset_complex_ppdb = get_path(
            '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src.sorted.rules')
        vocab_rules = get_path(
            '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.rules.sing')
        rule_size = 40246
    elif rule_base == 'v3':
        # Combine rules from v2 and single words from PPDB
        train_dataset_complex_ppdb = get_path(
            '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src.sorted.rules.v3')
        vocab_rules = get_path(
            '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.rules.v3.sing')
        rule_size = 55115
    else:
        # Original version of rules
        train_dataset_complex_ppdb = get_path(
            '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src.sorted.rules')
        vocab_rules = get_path(
            '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.rules')
        rule_size = 44186

    # add .dress extention will be same vocab as dress by add .dress in the end
    max_cand_rules = 15
    if args.our_vocab:
        vocab_simple = get_path('../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.dst.vocab')
        vocab_complex = get_path('../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src.vocab')
    else:
        vocab_simple = get_path('../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.dst.vocab.dress')
        vocab_complex = get_path('../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src.vocab.dress')
    # don't have dress version of tied vocab
    vocab_all = get_path('../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.vocab')
    if args.lower_case:
        vocab_simple = get_path(
            '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.dst.vocab.lower')
        vocab_complex = get_path(
            '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src.vocab.lower')
        vocab_all = vocab_all + '.lower'

    # Sub word config
    # if >0 using subword
    subword_vocab_size = args.subword_vocab_size
    subword_vocab_simple = get_path(
        '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.dst.vocab.subword.' + str(subword_vocab_size))
    subword_vocab_complex = get_path(
        '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src.vocab.subword.' + str(subword_vocab_size))
    subword_vocab_all = get_path(
        '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.vocab.subword.' + str(subword_vocab_size))


    # num_refs = 0
    # val_dataset_simple_folder = get_path('../text_simplification_data/train/dress/wikilarge/')
    # val_dataset_simple_file = 'wiki.full.aner.valid.dst'
    # val_dataset_complex = get_path('../text_simplification_data/train/dress/wikilarge/wiki.full.aner.valid.src')
    val_dataset_simple_folder = get_path('../text_simplification_data/val/')
    # use the original dress
    val_dataset_simple_file = 'wiki.full.aner.valid.dst'
    val_dataset_complex = get_path('../text_simplification_data/val/wiki.full.aner.valid.src')
    val_mapper = get_path('../text_simplification_data/val/tune.8turkers.tok.map.dress')
    # wiki.full.aner.ori.valid.dst is uppercase whereas tune.8turkers.tok.simp is lowercase
    val_dataset_simple_raw_file = 'wiki.full.aner.ori.valid.dst'
    val_dataset_complex_raw = get_path(
        '../text_simplification_data/val/wiki.full.aner.ori.valid.src')
    val_dataset_complex_rawlines_file = get_path(
        '../text_simplification_data/val/tune.8turkers.tok.norm')
    val_dataset_simple_rawlines_file_references = 'tune.8turkers.tok.turk.'
    val_dataset_simple_rawlines_file = 'tune.8turkers.tok.simp'
    num_refs = 8

    dimension = args.dimension

    max_complex_sentence = 85
    max_simple_sentence = 85
    if subword_vocab_size > 0:
        max_complex_sentence = 300
        max_simple_sentence = 300

    min_count = args.min_count
    batch_size = 32

    tokenizer = 'split'
    if dimension == 300:
        # We only have pretrained embedding for 300 dimension
        pretrained_embedding = get_path('../text_simplification_data/glove/glove.840B.300d.txt')
        pretrained_embedding_simple = get_path(
            '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.dst.vocab.pretrained')
        pretrained_embedding_complex = get_path(
            '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src.vocab.pretrained')


class WikiDressLargeTrainConfig(WikiDressLargeDefault):
    beam_search_size = 0


class WikiDressLargeTestConfig(WikiDressLargeDefault):
    beam_search_size = 1
    batch_size = 128
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
    # use the original dress
    val_dataset_simple_file = 'wiki.full.aner.valid.dst'
    val_dataset_complex = get_path('../text_simplification_data/val/wiki.full.aner.valid.src')
    val_mapper = get_path('../text_simplification_data/val/tune.8turkers.tok.map.dress')
    # wiki.full.aner.ori.valid.dst is uppercase whereas tune.8turkers.tok.simp is lowercase
    val_dataset_complex_rawlines_file = get_path(
        '../text_simplification_data/val/tune.8turkers.tok.norm')
    val_dataset_simple_rawlines_file_references = 'tune.8turkers.tok.turk.'
    val_dataset_simple_rawlines_file = 'tune.8turkers.tok.simp'
    num_refs = 8


class SubValWikiEightRefPPDBConfig(SubValWikiEightRefConfig):
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/eightref_val_ppdbe')
    ppdb_emode = 'weight'
    ppdb_emode_args = 1.5


class SubValWikiEightRefConfigBeam4(SubValWikiEightRefConfig):
    beam_search_size = 4
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/eightref_val_bm')


class SubTestWikiEightRefConfig(SubTest):
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/eightref_test', True)

    val_dataset_simple_folder = get_path('../text_simplification_data/test/')
    # use the original dress
    val_dataset_simple_file = 'wiki.full.aner.test.dst'
    val_dataset_complex = get_path('../text_simplification_data/test/wiki.full.aner.test.src')
    val_mapper = get_path('../text_simplification_data/test/test.8turkers.tok.map.dress')
    rule_base = args.rule_base
    if rule_base == 'v3':
        val_dataset_complex_ppdb = get_path('../text_simplification_data/test/wiki.full.aner.test.src.sorted.rules.v3')
    else:
        val_dataset_complex_ppdb = get_path(
            '../text_simplification_data/test/wiki.full.aner.test.src.sorted.rules')
    # wiki.full.aner.ori.test.dst is uppercase whereas test.8turkers.tok.simp is lowercase
    val_dataset_complex_rawlines_file = get_path(
        '../text_simplification_data/test/test.8turkers.tok.norm')
    val_dataset_simple_rawlines_file_references = 'test.8turkers.tok.turk.'
    val_dataset_simple_rawlines_file = 'test.8turkers.tok.simp'
    num_refs = 8

# Note that rule_base v3 is different from val_dataset_complex_ppdb v2
# rule_base v2 indicates a new set of rules by combining PPDB
# val_dataset_complex_ppdb v2 indicates revised (add X and NEW tag) rules for rule_base v1
class SubTestWikiEightRefConfigV2(SubTestWikiEightRefConfig):
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/eightref_test_v2', True)

    rule_base = args.rule_base
    if rule_base == 'v3':
        val_dataset_complex_ppdb = get_path('../text_simplification_data/test/wiki.full.aner.test.src.sorted.rules.v3')
    else:
        val_dataset_complex_ppdb = get_path(
        '../text_simplification_data/test/wiki.full.aner.test.src.sorted.rules.v2')


class SubTestWikiEightRefConfigV2Sing(SubTestWikiEightRefConfig):
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/eightref_test_v2sing', True)

    rule_base = args.rule_base
    if rule_base == 'v3':
        val_dataset_complex_ppdb = get_path('../text_simplification_data/test/wiki.full.aner.test.src.sorted.rules.v3')
    else:
        val_dataset_complex_ppdb = get_path(
        '../text_simplification_data/test/wiki.full.aner.test.src.sorted.rules.v2.sing')


class SubTestWikiEightRefPPDBConfig(SubTestWikiEightRefConfig):
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/eightref_test_ppdbe')
    ppdb_emode = 'weight'
    ppdb_emode_args = 1.5


class SubTestWikiEightRefPPDBConfigV2(SubTestWikiEightRefPPDBConfig):
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/eightref_test_ppdbe_v2', True)

    rule_base = args.rule_base
    if rule_base == 'v3':
        val_dataset_complex_ppdb = get_path('../text_simplification_data/test/wiki.full.aner.test.src.sorted.rules.v3')
    else:
        val_dataset_complex_ppdb = get_path(
        '../text_simplification_data/test/wiki.full.aner.test.src.sorted.rules.v2')


class SubTestWikiEightRefPPDBConfigV2Sing(SubTestWikiEightRefPPDBConfig):
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/eightref_test_ppdbe_v2sing', True)

    rule_base = args.rule_base
    if rule_base == 'v3':
        val_dataset_complex_ppdb = get_path('../text_simplification_data/test/wiki.full.aner.test.src.sorted.rules.v3')
    else:
        val_dataset_complex_ppdb = get_path(
        '../text_simplification_data/test/wiki.full.aner.test.src.sorted.rules.v2.sing')


class SubTestWikiEightRefConfigBeam4(SubTestWikiEightRefConfig):
    beam_search_size = 4
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/eightref_test_bm')


class WikiDressSmallDefault(WikiDressLargeDefault):
    train_dataset_simple = get_path('../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.train.dst')
    train_dataset_simple_ppdb = get_path(
        '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.train.dst.rules')
    train_dataset_simple_syntax = get_path(
        '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.train.dst.jsyntax')
    train_dataset_complex_syntax = get_path(
        '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.train.src.jsyntax')
    train_dataset_complex = get_path('../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.train.src')
    # ../ text_simplification_data / train / dress / wikilarge / wiki.full.aner.train.src.rules
    train_dataset_complex_ppdb = get_path(
        '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.train.src.rules')
    vocab_simple = get_path('../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.train.dst.vocab')
    vocab_complex = get_path('../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.train.src.vocab')


class WikiDressSmallTrainConfig(WikiDressSmallDefault):
    beam_search_size = 0


class SubTestWikiSmallConfig(WikiDressSmallDefault):
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/small_test')

    val_dataset_simple_folder = get_path(
        '../text_simplification_data/train/dress/wikismall/')
    # use the original dress
    val_dataset_simple_file = 'PWKP_108016.tag.80.aner.test.dst'
    val_dataset_complex = get_path(
        '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.test.src')
    val_mapper = get_path('../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.ori.test.map')
    # wiki.full.aner.ori.test.dst is uppercase whereas test.8turkers.tok.simp is lowercase
    val_dataset_complex_rawlines_file = get_path(
        '../text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.ori.test.src')
    val_dataset_simple_rawlines_file = 'PWKP_108016.tag.80.aner.ori.test.dst'
    num_refs = 0
    beam_search_size = 1


class SubTestWikiSmallPPDBConfig(SubTestWikiSmallConfig):
    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/small_test_ppdbe')
    ppdb_emode = 'weight'
    ppdb_emode_args = 1.5


""" 
==========================================================================================
For experiment after Jan 2018
==========================================================================================
"""

class WikiTransLegacyBaseCfg(WikiDressLargeDefault):
    train_dataset_complex_syntax = get_path(
        '../text_simplification_data/train/dress/wikilarge2/src.jsyntax.txt')
    train_dataset_complex_ppdb = get_path(
        '../text_simplification_data/train/dress/wikilarge2/src.sorted.rules.txt')
    vocab_rules = get_path(
        '../text_simplification_data/train/dress/wikilarge2/rules.txt')
    train_dataset_simple = get_path('../text_simplification_data/train/dress/wikilarge2/dst.txt')
    train_dataset_complex = get_path('../text_simplification_data/train/dress/wikilarge2/src.txt')
    batch_size = args.batch_size
    rule_size = 72445
    max_cand_rules = 15

class WikiTransLegacyTrainCfg(WikiTransLegacyBaseCfg):
    beam_search_size = 0
    number_samples = args.number_samples


class WikiTransLegacyTestCfg(WikiTransLegacyBaseCfg):
    batch_size = 64
    replace_unk_by_emb = True
    beam_search_size = 1

    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/eightref_test', True)

    val_dataset_simple_folder = get_path('../text_simplification_data/test/')
    # use the original dress
    val_dataset_simple_file = 'wiki.full.aner.test.dst'
    val_dataset_complex = get_path('../text_simplification_data/test/wiki.full.aner.test.src')
    val_mapper = get_path('../text_simplification_data/test/test.8turkers.tok.map.dress')
    # wiki.full.aner.ori.test.dst is uppercase whereas test.8turkers.tok.simp is lowercase
    val_dataset_complex_rawlines_file = get_path(
        '../text_simplification_data/test/test.8turkers.tok.norm')
    val_dataset_simple_rawlines_file_references = 'test.8turkers.tok.turk.'
    val_dataset_simple_rawlines_file = 'test.8turkers.tok.simp'
    num_refs = 8
    val_dataset_complex_syntax = get_path(
        '../text_simplification_data/test/wiki.full.aner.test.src.jsyntax')
    val_dataset_complex_ppdb = get_path(
        '../text_simplification_data/test/wiki.full.aner.test.src.sorted.rules')

class WikiTransBaseCfg(DefaultConfig):
    model_print_freq = 50
    save_model_secs = 600
    model_eval_freq = args.model_eval_freq

    train_dataset_simple = get_path('../text_simplification_data/wiki/ner3/ner_simp.txt')
    # train_dataset_simple_ppdb = get_path(
    #     '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.dst.rules')
    # train_dataset_simple_syntax = get_path(
    #     '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.dst.jsyntax')
    # train_dataset_complex_syntax = get_path(
    #     '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src.jsyntax')
    train_dataset_complex = get_path('../text_simplification_data/wiki/ner3/ner_comp.txt')
    # train_dataset_complex_ppdb = get_path(args.train_dataset_complex_ppdb)
    subword_vocab_size = args.subword_vocab_size
    if subword_vocab_size == 30000:
        vocab_simple = get_path('../text_simplification_data/wiki/voc/voc_simp_sub30k.txt')
        vocab_complex = get_path('../text_simplification_data/wiki/voc/voc_comp_sub30k.txt')
        vocab_all = get_path('../text_simplification_data/wiki/voc/voc_all_sub30k.txt')
    else:
        vocab_simple = get_path('../text_simplification_data/wiki/voc/voc_simp.txt')
        vocab_complex = get_path('../text_simplification_data/wiki/voc/voc_comp.txt')
        vocab_all = get_path('../text_simplification_data/wiki/voc/voc_all.txt')

    val_dataset_simple_folder = get_path('../text_simplification_data/val/')
    # use the original dress
    val_dataset_simple_file = 'wiki.full.aner.valid.dst'
    val_dataset_complex = get_path('../text_simplification_data/val/wiki.full.aner.valid.src')
    val_mapper = get_path('../text_simplification_data/val/tune.8turkers.tok.map.dress')
    # wiki.full.aner.ori.valid.dst is uppercase whereas tune.8turkers.tok.simp is lowercase
    val_dataset_simple_raw_file = 'wiki.full.aner.ori.valid.dst'
    val_dataset_complex_raw = get_path(
        '../text_simplification_data/val/wiki.full.aner.ori.valid.src')
    val_dataset_complex_rawlines_file = get_path(
        '../text_simplification_data/val/tune.8turkers.tok.norm')
    val_dataset_simple_rawlines_file_references = 'tune.8turkers.tok.turk.'
    val_dataset_simple_rawlines_file = 'tune.8turkers.tok.simp'
    num_refs = 8

    dimension = args.dimension

    max_complex_sentence = 85
    max_simple_sentence = 85

    min_count = 50
    batch_size = args.batch_size

    tokenizer = 'split'


class WikiTransTrainCfg(WikiTransBaseCfg):
    beam_search_size = 0
    number_samples = args.number_samples


class WikiTransValCfg(WikiTransBaseCfg):
    batch_size = 64
    replace_unk_by_emb = True
    beam_search_size = 1

    output_folder = args.output_folder
    resultdor = get_path('../' + output_folder + '/result/eightref_test')

    val_dataset_simple_folder = get_path('../text_simplification_data/test/')
    # use the original dress
    val_dataset_simple_file = 'wiki.full.aner.test.dst'
    val_dataset_complex = get_path('../text_simplification_data/test/wiki.full.aner.test.src')
    val_mapper = get_path('../text_simplification_data/test/test.8turkers.tok.map.dress')
    # wiki.full.aner.ori.test.dst is uppercase whereas test.8turkers.tok.simp is lowercase
    val_dataset_complex_rawlines_file = get_path(
        '../text_simplification_data/test/test.8turkers.tok.norm')
    val_dataset_simple_rawlines_file_references = 'test.8turkers.tok.turk.'
    val_dataset_simple_rawlines_file = 'test.8turkers.tok.simp'
    num_refs = 8


def list_config(config):
    attrs = [attr for attr in dir(config)
               if not callable(getattr(config, attr)) and not attr.startswith("__")]
    output = ''
    for attr in attrs:
        val = getattr(config, attr)
        output = '\n'.join([output, '%s=\t%s' % (attr, val)])
    return output