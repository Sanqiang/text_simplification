import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Model Parameter')
    parser.add_argument('-fw', '--framework', default='transformer',
                        help='Framework we are using?')
    parser.add_argument('-out', '--output_folder', default='tmp',
                        help='Output folder?')
    parser.add_argument('-warm', '--warm_start', default='',
                        help='Path for warm start checkpoint?')
    parser.add_argument('-upr', '--use_partial_restore', default=True, type=bool,
                        help='Whether to use partial restore?')
    parser.add_argument('-cpu', '--use_cpu', default=False, type=bool,
                        help='Whether to use cpu for large memory part, e.g. embedding?')

    parser.add_argument('-mode', '--mode', default='dress',
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
    parser.add_argument('-lc', '--lower_case', default=False, type=bool,
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

    # For Graph
    parser.add_argument('-dim', '--dimension', default=300, type=int,
                        help='Size of dimension?')
    parser.add_argument('-emb', '--tied_embedding', default='none',
                        help='Version of tied embedding?')
    parser.add_argument('-loss', '--loss_fn', default='sampled_softmax',
                        help='Loss function?')
    parser.add_argument('-ns', '--number_samples', default=10, type=int,
                        help='Number of samples used in Softmax?')
    parser.add_argument('-uqm', '--use_quality_model', default=False, type=bool,
                        help='Whether to use quality model?')
    parser.add_argument('-decay', '--use_learning_rate_decay', default=False, type=bool,
                        help='Whether to use learning rate decay?')

    parser.add_argument('-rl_len', '--rl_prelenth', default=3, type=int,
                        help='Length of output sentences before RL applied?')
    parser.add_argument('-rl_bleu', '--rl_bleu', default=0.0, type=float,
                        help='The weight for BLEU in RL?')
    parser.add_argument('-rl_sari', '--rl_sari', default=0.0, type=float,
                        help='The weight for SARI in RL?')
    parser.add_argument('-rl_fkgl', '--rl_fkgl', default=0.0, type=float,
                        help='The weight for FKGL in RL?')
    # For PPDB
    parser.add_argument('-rl_simp', '--rl_simp', default=0.0, type=float,
                        help='The weight for SARI simple in RL?')
    parser.add_argument('-rl_keep', '--rl_keep', default=0.0, type=float,
                        help='The weight for SARI keep in RL?')

    parser.add_argument('-path_ppdb', '--path_ppdb_refine',
                        default='../text_simplification_data/ppdb/SimplePPDB.enrich',
                        help='The path for PPDB rules?')
    parser.add_argument('-complex_ppdb', '--train_dataset_complex_ppdb',
                        default='../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src.rules',
                        help='The path for PPDB comp rules?')


    # For Transformer
    parser.add_argument('-pos', '--hparams_pos', default='timing',
                        help='Whether to use positional encoding?')
    parser.add_argument('-digate', '--decode_input_gate', default=False,
                        help='Whether to use gate in decode input?')
    parser.add_argument('-dagate', '--decode_atten_gate', default=False,
                        help='Whether to use gate in decode attention?')
    parser.add_argument('-tlg', '--trans_layer_gate', default=False,
                        help='Whether to use gate in each layer of Transformer?')
    parser.add_argument('-eaf', '--encoder_attn_flatten', default=False,
                        help='Whether to use each layer output from encoder?')

    parser.add_argument('-nhl', '--num_hidden_layers', default=6, type=int,
                        help='Number of hidden layer?')
    parser.add_argument('-nel', '--num_encoder_layers', default=6, type=int,
                        help='Number of encoder layer?')
    parser.add_argument('-ndl', '--num_decoder_layers', default=6, type=int,
                        help='Number of decoder layer?')
    parser.add_argument('-nh', '--num_heads', default=5, type=int,
                        help='Number of multi-attention heads?')
    parser.add_argument('-penalty_alpha', '--penalty_alpha', default=0.6, type=float,
                        help='The alpha for length penalty?')
    parser.add_argument('-aloss', '--attn_loss', default='',
                        help='Attention Reconstruction loss config?')

    # For Experiment
    parser.add_argument('-exp_penalty_alpha', '--exp_penalty_alpha', default=False,
                        help='Whether to do penalty alpha experiment')



    # For Test
    parser.add_argument('-test_ckpt', '--test_ckpt', default='',
                        help='Path for test ckpt checkpoint?')

    args = parser.parse_args()
    return args