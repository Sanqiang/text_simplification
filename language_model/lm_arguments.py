from model.model_config import get_path
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Model Parameter')

    parser.add_argument('-ngpus', '--num_gpus', default=1, type=int,
                        help='Number of GPU cards?')
    parser.add_argument('-bsize', '--batch_size', default=64, type=int,
                        help='Size of Mini-Batch?')
    parser.add_argument('-dim', '--dimension', default=512, type=int,
                        help='Size of dimension?')
    parser.add_argument('-maxlen', '--max_sent_len', default=100, type=int,
                        help='Max of sentence length?')
    parser.add_argument('-mc', '--min_count', default=50, type=int,
                        help='Truncate the vocabulary less than equal to the count?')
    parser.add_argument('-lr', '--learning_rate', default=0.1, type=float,
                        help='Value of learning rate?')

    parser.add_argument('-vocab_path', '--vocab_path',
                        default=get_path('../text_simplification_data/wiki/voc/voc_all.txt'),
                        help='The path for Vocab file?')
    parser.add_argument('-data_path', '--data_path',
                        default=get_path('../text_simplification_data/lm_data/'),
                        help='The path for Data folder?')
    parser.add_argument('-logdir', '--logdir',
                        default=get_path('../lm/log/'),
                        help='The path for Log?')

    parser.add_argument('-nh', '--num_heads', default=5, type=int,
                        help='Number of heads?')
    parser.add_argument('-nhl', '--num_hidden_layers', default=6, type=int,
                        help='Number of hidden layer?')
    parser.add_argument('-ldropout', '--layer_prepostprocess_dropout', default=0.2, type=float,
                        help='Dropout rate?')

    args = parser.parse_args()
    return args