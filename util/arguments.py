import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Model Parameter')
    parser.add_argument('-fw', '--framework', default='transformer',
                        help='Framework we are using?')
    parser.add_argument('-uqm', '--use_quality_model', default=False, type=bool,
                        help='Whether to use quality model?')
    parser.add_argument('-out', '--output_folder', default='tmp',
                        help='Output folder?')
    parser.add_argument('-pos', '--hparams_pos', default='timing',
                        help='Whether to use positional encoding?')
    parser.add_argument('-warm', '--warm_start', default='',
                        help='Path for warm start checkpoint?')

    parser.add_argument('-mode', '--mode', default='dress',
                        help='The Usage Model?')
    args = parser.parse_args()
    return args