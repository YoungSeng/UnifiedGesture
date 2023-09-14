import argparse
'''
pip install tensorboard

'''

dataset_name = 'Mixamo_new_2'       # modify for test

multiple = 1

def get_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--save_dir', type=str, default='./pretrained', help='directory for all savings')       # modify 3
    # parser.add_argument('--save_dir', type=str, default='./training', help='directory for all savings')
    # modify for test
    parser.add_argument('--save_dir', type=str, default='./my_model_new_3', help='directory for all savings')
    parser.add_argument('--cuda_device', type=str, default='cuda:5', help='cuda device number, eg:[cuda:0]')

    parser.add_argument('--num_layers', type=int, default=2, help='number of layers')       # model_5, model_7
    parser.add_argument('--learning_rate', type=float, default=multiple * 2e-4, help='learning rate')
    parser.add_argument('--alpha', type=float, default=0, help='penalty of sparsity')
    parser.add_argument('--batch_size', type=int, default=256 * multiple, help='batch_size')     # 256
    parser.add_argument('--upsampling', type=str, default='linear', help="'stride2' or 'nearest', 'linear'")
    parser.add_argument('--downsampling', type=str, default='stride2', help='stride2 or max_pooling')
    parser.add_argument('--batch_normalization', type=int, default=0, help='batch_norm: 1 or 0')
    parser.add_argument('--activation', type=str, default='LeakyReLU', help='activation: ReLU, LeakyReLU, tanh')
    parser.add_argument('--rotation', type=str, default='quaternion', help='representation of rotation:euler_angle, quaternion')
    parser.add_argument('--data_augment', type=int, default=1, help='data_augment: 1 or 0')
    parser.add_argument('--epoch_num', type=int, default=20001, help='epoch_num')
    parser.add_argument('--window_size', type=int, default=64, help='length of time axis per window')
    parser.add_argument('--kernel_size', type=int, default=15, help='must be odd')
    parser.add_argument('--base_channel_num', type=int, default=-1)
    parser.add_argument('--normalization', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--skeleton_dist', type=int, default=2)
    parser.add_argument('--skeleton_pool', type=str, default='mean')
    parser.add_argument('--extra_conv', type=int, default=0)
    parser.add_argument('--padding_mode', type=str, default='reflection')
    parser.add_argument('--dataset_constant', type=str, default=dataset_name)
    parser.add_argument('--dataset', type=str, default=dataset_name)
    parser.add_argument('--fk_world', type=int, default=0)
    parser.add_argument('--patch_gan', type=int, default=1)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--skeleton_info', type=str, default='concat')
    parser.add_argument('--ee_loss_fact', type=str, default='height')
    parser.add_argument('--pos_repr', type=str, default='3d')
    parser.add_argument('--D_global_velo', type=int, default=0)
    parser.add_argument('--gan_mode', type=str, default='lsgan')
    parser.add_argument('--pool_size', type=int, default=50)
    parser.add_argument('--is_train', type=int, default=1)
    parser.add_argument('--model', type=str, default='mul_top_mul_ske')
    parser.add_argument('--epoch_begin', type=int, default=0)
    parser.add_argument('--lambda_rec', type=float, default=5)
    parser.add_argument('--lambda_cycle', type=float, default=5)
    parser.add_argument('--lambda_ee', type=float, default=100)
    parser.add_argument('--lambda_global_pose', type=float, default=2.5)
    parser.add_argument('--lambda_position', type=float, default=1)
    parser.add_argument('--ee_velo', type=int, default=1)
    parser.add_argument('--ee_from_root', type=int, default=1)
    parser.add_argument('--scheduler', type=str, default='none')
    parser.add_argument('--rec_loss_mode', type=str, default='extra_global_pos')
    parser.add_argument('--adaptive_ee', type=int, default=0)
    parser.add_argument('--simple_operator', type=int, default=0)
    parser.add_argument('--use_sep_ee', type=int, default=0)
    parser.add_argument('--eval_seq', type=int, default=0)

    parser.add_argument('--model_name', type=str, default='diffusion')
    parser.add_argument('--target', type=str, default='ZEGGS')
    parser.add_argument('--input_file', type=str, default="../diffusion_latent/result_model3_128/Trinity/005_Neutral_4_x_1_0_minibatch_30_[0, 0, 0, 0, 0, 3, 0]_123456_recon.npy")
    parser.add_argument('--ref_path', type=str, default='./datasets/bvh2latent/ZEGGS/065_Speech_0_x_1_0.npy')
    parser.add_argument('--output_path', type=str, default='../result/inference/Trinity/')

    return parser


def get_args():
    parser = get_parser()
    return parser.parse_args()


def get_std_bvh(args=None, dataset=None):
    if args is None and dataset is None: raise Exception('Unexpected parameter')
    if dataset is None: dataset = args.dataset
    std_bvh = './datasets/' + dataset_name + '/std_bvhs/{}.bvh'.format(dataset)
    return std_bvh


def try_mkdir(path):
    import os
    if not os.path.exists(path):
        os.system('mkdir -p {}'.format(path))
