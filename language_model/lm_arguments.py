from model.model_config import get_path
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='LM Model Parameter')

    parser.add_argument('-out', '--output_folder', default='tmp',
                        help='Output folder?')
    parser.add_argument('-ngpus', '--num_gpus', default=1, type=int,
                        help='Number of GPU cards?')
    parser.add_argument('-bsize', '--batch_size', default=32, type=int,
                        help='Size of Mini-Batch?')
    parser.add_argument('-dim', '--dimension', default=300, type=int,
                        help='Size of dimension?')
    parser.add_argument('-svoc_size', '--subword_vocab_size', default=0, type=int,
                        help='Size of subword?')
    parser.add_argument('-ns', '--number_samples', default=0, type=int,
                        help='Size of samples for sampled softmax?')
    parser.add_argument('-maxlen', '--max_sent_len', default=100, type=int,
                        help='Max of sentence length?')
    parser.add_argument('-mc', '--min_count', default=5, type=int,
                        help='Truncate the vocabulary less than equal to the count?')
    parser.add_argument('-lr', '--learning_rate', default=0.1, type=float,
                        help='Value of learning rate?')
    parser.add_argument('-vocab_path', '--vocab_path',
                        default=get_path(
                        '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src.vocab.lower'), #get_path('../text_simplification_data/wiki/voc/voc_all.txt'),
                        help='The path for Vocab file?')
    parser.add_argument('-data_path', '--data_path',
                        default=get_path('../text_simplification_data/lm_data/'),
                        help='The path for Data folder?')
    parser.add_argument('-evaldata_path', '--evaldata_path',
                        default=get_path('../text_simplification_data/val/tune.8turkers.tok.norm.ner'),
                        help='The path for Eval Data?')
    parser.add_argument('-logdir', '--logdir',
                        default=get_path('../lm/log/', True),
                        help='The path for Log?')
    parser.add_argument('-resultdir', '--resultdir',
                        default=get_path('../lm/result/', True),
                        help='The path for Result?')
    parser.add_argument('-modeldir', '--modeldir',
                        default=get_path('../lm/model/', True),
                        help='The path for Model?')
    parser.add_argument('-nh', '--num_heads', default=5, type=int,
                        help='Number of heads?')
    parser.add_argument('-nhl', '--num_hidden_layers', default=6, type=int,
                        help='Number of hidden layer?')
    parser.add_argument('-ldropout', '--layer_prepostprocess_dropout', default=0.2, type=float,
                        help='Dropout rate?')

    args = parser.parse_args()
    return postprocess_args(args)


def postprocess_args(args):
    output_folder = args.output_folder
    args.logdir = args.logdir.replace('lm', output_folder)
    args.resultdir = args.resultdir.replace('lm', output_folder)
    args.modeldir = args.modeldir.replace('lm', output_folder)
    if args.subword_vocab_size > 0:
        args.vocab_path = get_path('../text_simplification_data/wiki/voc/voc_all_sub50k.txt')
    return args