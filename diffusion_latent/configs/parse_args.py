import configargparse
import argparse

def str2bool(v):
    """ from https://stackoverflow.com/a/43357954/1361529 """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def parse_args():

    parser = argparse.ArgumentParser(description='UnifiedGesture')
    parser.add_argument('--config', default='./configs/codebook.yml')

    # UnifiedGesture
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--no_cuda', type=list, default=['2'])
    parser.add_argument('--save_dir', type=str, default='./save')
    parser.add_argument('--model_path', type=str, default='./experiments')
    parser.add_argument('--audio_path', type=str, default='./experiments')

    args = parser.parse_args()
    args.no_cuda = args.gpu
    return args
