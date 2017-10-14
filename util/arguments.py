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

    # For Graph
    parser.add_argument('-emb', '--tied_embedding', default='none',
                        help='Version of tied embedding?')
    parser.add_argument('-loss', '--loss_fn', default='sampled_softmax',
                        help='Loss function?')
    parser.add_argument('-ns', '--number_samples', default=10, type=int,
                        help='Number of samples used in Softmax?')
    parser.add_argument('-uqm', '--use_quality_model', default=False, type=bool,
                        help='Whether to use quality model?')

    # For Transformer
    parser.add_argument('-pos', '--hparams_pos', default='timing',
                        help='Whether to use positional encoding?')
    parser.add_argument('-digate', '--decode_input_gate', default=False,
                        help='Whether to use gate in decode input?')
    parser.add_argument('-dagate', '--decode_atten_gate', default=False,
                        help='Whether to use gate in decode attention?')
    parser.add_argument('-nhl', '--num_hidden_layers', default=6, type=int,
                        help='Number of hidden layer?')
    parser.add_argument('-nel', '--num_encoder_layers', default=6, type=int,
                        help='Number of encoder layer?')
    parser.add_argument('-ndl', '--num_decoder_layers', default=6, type=int,
                        help='Number of decoder layer?')
    parser.add_argument('-nh', '--num_heads', default=5, type=int,
                        help='Number of multi-attention heads?')



    args = parser.parse_args()
    return args