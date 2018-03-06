import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Model Parameter')
    parser.add_argument('-ngpus', '--num_gpus', default=1, type=int,
                        help='Number of GPU cards?')
    parser.add_argument('-bsize', '--batch_size', default=128, type=int,
                        help='Size of Mini-Batch?')
    parser.add_argument('-fw', '--framework', default='transformer',
                        help='Framework we are using?')
    parser.add_argument('-env', '--environment', default='crc',
                        help='The environment machine?')
    parser.add_argument('-out', '--output_folder', default='tmp',
                        help='Output folder?')
    parser.add_argument('-warm', '--warm_start', default='',
                        help='Path for warm start checkpoint?')
    parser.add_argument('-upr', '--use_partial_restore', default=True, type=bool,
                        help='Whether to use partial restore?')
    parser.add_argument('-tmode', '--train_mode', default='teacher',
                        help='The mode of training?')

    parser.add_argument('-mode', '--mode', default='wiki',
                        help='The Usage Model?')
    parser.add_argument('-op', '--optimizer', default='adagrad',
                        help='Which optimizer to use?')
    parser.add_argument('-lr', '--learning_rate', default=0.1, type=float,
                        help='Value of learning rate?')
    parser.add_argument('-layer_drop', '--layer_prepostprocess_dropout', default=0.0, type=float,
                        help='Dropout rate for data input?')
    parser.add_argument('-cop', '--change_optimizer', default=False, type=bool,
                        help='Whether to change the optimizer?')

    # For Data
    parser.add_argument('-lc', '--lower_case', default=True, type=bool,
                        help='Whether to lowercase the vocabulary?')
    parser.add_argument('-mc', '--min_count', default=5, type=int,
                        help='Truncate the vocabulary less than equal to the count?')
    parser.add_argument('-ovoc', '--our_vocab', default=False, type=bool,
                        help='Whether to use our own vocab')
    parser.add_argument('-ppdb', '--ppdb_mode', default='none',
                        help='PPDB mode?')
    parser.add_argument('-ppdbw', '--ppdb_args', default=None,
                        help='PPDB arguments?')
    parser.add_argument('-ppdbe', '--ppdb_emode', default='none',
                        help='PPDB mode for eval?')
    parser.add_argument('-ppdbew', '--ppdb_emode_args', default=None,
                        help='PPDB eval arguments?')
    parser.add_argument('-svoc_size', '--subword_vocab_size', default=0, type=int,
                        help='The size of subword vocabulary? if <= 0, not use subword unit.')
    parser.add_argument('-eval_freq', '--model_eval_freq', default=10000, type=int,
                        help='The frequency of evaluation at training? not use if = 0.')
    parser.add_argument('-itrain', '--it_train', default=False, type=bool,
                        help='Whether to iterate train data set?')
    parser.add_argument('-rdec', '--rnn_decoder', default=False, type=bool,
                        help='Whether to RNN decoder?')
    parser.add_argument('-dataset2', '--use_dataset2', default=False, type=bool,
                        help='Whether to use second dataset with iteractively?')

    # For Graph
    parser.add_argument('-dim', '--dimension', default=300, type=int,
                        help='Size of dimension?')
    parser.add_argument('-emb', '--tied_embedding', default='none',
                        help='Version of tied embedding?')
    parser.add_argument('-ns', '--number_samples', default=0, type=int,
                        help='Number of samples used in Softmax?')

    parser.add_argument('-path_ppdb', '--path_ppdb_refine',
                        default='../text_simplification_data/ppdb/SimplePPDB.enrich',
                        help='The path for PPDB rules?')

    # For Memory
    parser.add_argument('-mem', '--memory', default=None,
                        help='Separate memory?')
    parser.add_argument('-memcfg', '--memory_config', default='',
                        help='Memory Config?')
    parser.add_argument('-memstep', '--memory_prepare_step', default=300, type=int,
                        help='Number of steps for memory prepare?')

    # For RL
    parser.add_argument('-rlcfg', '--rl_config', default='',
                        help='reinforce learning Config?')

    # For Transformer
    parser.add_argument('-pos', '--hparams_pos', default='timing',
                        help='Whether to use positional encoding?')

    parser.add_argument('-nhl', '--num_hidden_layers', default=4, type=int,
                        help='Number of hidden layer?')
    parser.add_argument('-nel', '--num_encoder_layers', default=4, type=int,
                        help='Number of encoder layer?')
    parser.add_argument('-ndl', '--num_decoder_layers', default=4, type=int,
                        help='Number of decoder layer?')
    parser.add_argument('-nh', '--num_heads', default=5, type=int,
                        help='Number of multi-attention heads?')
    parser.add_argument('-penalty_alpha', '--penalty_alpha', default=0.6, type=float,
                        help='The alpha for length penalty?')

    # For Test
    parser.add_argument('-test_ckpt', '--test_ckpt', default='',
                        help='Path for test ckpt checkpoint?')

    parser.add_argument('-rbase', '--rule_base', default='v1',
                        help='Which rule base to use?')

    # Useless
    parser.add_argument('-maxlen', '--max_sent_len', default=100, type=int,
                        help='Max of sentence length?')

    args = parser.parse_args()
    return args