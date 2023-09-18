import os
import pdb

import numpy as np

import pdb
import yaml
from pprint import pprint
from easydict import EasyDict
import torch
from configs.parse_args import parse_args
import glob
from models.vqvae import VQVAE_ulr, VQVAE_ulr_2
import torch.nn as nn

args = parse_args()
mydevice = torch.device('cuda:' + args.gpu)

with open(args.config) as f:
    config = yaml.safe_load(f)

for k, v in vars(args).items():
    config[k] = v
pprint(config)

config = EasyDict(config)

config.no_cuda = config.gpu


def main(upper, lower, root, model, save_path, prefix, model_name='vqvae_ulr_2'):

    with torch.no_grad():
        result_upper = []
        result_lower = []
        result_root = []

        x1 = torch.tensor(upper).unsqueeze(0).to(mydevice)
        x2 = torch.tensor(lower).unsqueeze(0).to(mydevice)
        x3 = torch.tensor(root).unsqueeze(0).to(mydevice)

        if model_name == 'vqvae_ulr':
            # zs1, zs2, zs3 = model.module.get_code(x1.float(), x2.float(), x3.float())
            zs1, _ = model.module.get_code(x1.float())
        elif model_name == 'vqvae_ulr_2':
            x2 = torch.cat([x2, x3], dim=2)
            zs1, zs2 = model.module.get_code(x1.float(), x2.float())

        if model_name == 'vqvae_ulr':
            # y1, y2, y3 = model.module.code_2_pose(zs1, zs2, zs3)
            y1 = model.module.code_2_pose(zs1)
        elif model_name == 'vqvae_ulr_2':
            y1, y2 = model.module.code_2_pose(zs1, zs2)

        result_upper.append(y1.squeeze(0).data.cpu().numpy())
        # result_lower.append(y2.squeeze(0).data.cpu().numpy())
        # if model_name == 'vqvae_ulr':
        #     result_root.append(y3.squeeze(0).data.cpu().numpy())

    out_zs1 = np.vstack(zs1[0].squeeze(0).data.cpu().numpy())
    # out_zs2 = np.vstack(zs2[0].squeeze(0).data.cpu().numpy())
    # if model_name == 'vqvae_ulr':
    #     out_zs3 = np.vstack(zs3[0].squeeze(0).data.cpu().numpy())
    #     out_root = np.vstack(result_root)
    out_upper = np.vstack(result_upper)
    # out_lower = np.vstack(result_lower)

    # if model_name == 'vqvae_ulr':
    #     print(out_upper.shape, out_lower.shape, out_root.shape)
    # elif model_name == 'vqvae_ulr_2':
    #     print(out_upper.shape, out_lower.shape)

    np.save(os.path.join(save_path, 'code_upper_' + prefix + '.npy'), out_zs1)
    # np.save(os.path.join(save_path, 'code_lower_' + prefix + '.npy'), out_zs2)
    np.save(os.path.join(save_path, 'generate_upper_' + prefix + '.npy'), np.expand_dims(out_upper.transpose(), axis=0))
    # np.save(os.path.join(save_path, 'generate_lower_' + prefix + '.npy'), np.expand_dims(out_lower.transpose(), axis=0))
    if model_name == 'vqvae_ulr':
        # np.save(os.path.join(save_path, 'code_root_' + prefix + '.npy'), out_zs3)
        # np.save(os.path.join(save_path, 'generate_root_' + prefix + '.npy'), np.expand_dims(out_root.transpose(), axis=0))
        # return np.expand_dims(out_upper.transpose(), axis=0), np.expand_dims(out_lower.transpose(), axis=0), \
        #        np.expand_dims(out_root.transpose(), axis=0)
        np.expand_dims(out_upper.transpose(), axis=0)

    elif model_name == 'vqvae_ulr_2':
        return np.expand_dims(out_upper.transpose(), axis=0), np.expand_dims(out_lower.transpose(), axis=0)


def main_2(upper, model, save_path, prefix, w_grad, code=None):
    if not w_grad:
        with torch.no_grad():
            result_upper = []
            x1 = torch.tensor(upper).unsqueeze(0).to(mydevice)
            if type(code) is np.ndarray:
                print('use code')
                zs1 = [torch.tensor(code).to(mydevice)]
                out_distance = None
            else:
                zs1, distance = model.module.get_code(x1.float())
                out_distance = np.vstack(distance[0].squeeze(0).data.cpu().numpy())

            y1 = model.module.code_2_pose(zs1)
            result_upper.append(y1.squeeze(0).data.cpu().numpy())
    else:
        result_upper = []

        x1 = upper.unsqueeze(0).to(mydevice)
        zs1, distance = model.module.get_code(x1.float())
        out_distance = torch.cat(distance, dim=0)
        y1 = model.module.code_2_pose(zs1)
        result_upper.append(y1.squeeze(0).data.cpu().numpy())

    out_zs1 = np.vstack(zs1[0].squeeze(0).data.cpu().numpy())
    out_upper = np.vstack(result_upper)
    # print(out_upper.shape)
    # np.save(os.path.join(save_path, 'code_upper_' + prefix + '.npy'), out_zs1)
    # np.save(os.path.join(save_path, 'generate_upper_' + prefix + '.npy'), np.expand_dims(out_upper.transpose(), axis=0))
    return np.expand_dims(out_upper.transpose(), axis=0), out_zs1, out_distance



def visualize_code(model, save_path, prefix, code_upper, code_lower=None, code_root=None, model_name='vqvae_ulr_2'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with torch.no_grad():
        result_upper = []
        result_lower = []
        if model_name == 'vqvae_ulr':
            result_root = []
            code_root = torch.tensor(code_root).unsqueeze(0).to(mydevice)

        code_upper = torch.tensor(code_upper).unsqueeze(0).to(mydevice)
        if code_lower:
            code_lower = torch.tensor(code_lower).unsqueeze(0).to(mydevice)

        if model_name == 'vqvae_ulr':
            y1, y2, y3 = model.module.code_2_pose([code_upper], [code_lower], [code_root])
            result_root.append(y3.squeeze(0).data.cpu().numpy())
            out_root = np.vstack(result_root)

        elif model_name == 'vqvae_ulr_2':
            if code_lower:
                y1, y2 = model.module.code_2_pose([code_upper], [code_lower])
            else:
                y1 = model.module.code_pose([code_upper])

        if code_lower:
            result_lower.append(y2.squeeze(0).data.cpu().numpy())
        result_upper.append(y1.squeeze(0).data.cpu().numpy())


    out_upper = np.vstack(result_upper)
    if code_lower:
        out_lower = np.vstack(result_lower)

    if model_name == 'vqvae_ulr':
        print(out_upper.shape, out_lower.shape, out_root.shape)
        np.save(os.path.join(save_path, 'generate_root_' + prefix), np.expand_dims(out_root.transpose(), axis=0))
    elif model_name == 'vqvae_ulr_2':
        if code_lower:
            print(out_upper.shape, out_lower.shape)
            np.save(os.path.join(save_path, 'generate_lower_' + prefix), np.expand_dims(out_lower.transpose(), axis=0))
        else:
            print(out_upper.shape)
        np.save(os.path.join(save_path, 'generate_upper_' + prefix), np.expand_dims(out_upper.transpose(), axis=0))

    if model_name == 'vqvae_ulr':
        return np.expand_dims(out_upper.transpose(), axis=0), np.expand_dims(out_lower.transpose(), axis=0), \
           np.expand_dims(out_root.transpose(), axis=0)
    elif model_name == 'vqvae_ulr_2':
        if code_lower:
            return np.expand_dims(out_upper.transpose(), axis=0), np.expand_dims(out_lower.transpose(), axis=0)
        else:
            return np.expand_dims(out_upper.transpose(), axis=0)


def fold2code(path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    upper_files = sorted(glob.glob(path + "/*_upper.npy"))
    for item in upper_files:
        path = os.path.split(item)[0]
        name = os.path.split(item)[1]
        source_upper = np.load(os.path.join(path, name))[0].transpose()
        source_lower = np.load(os.path.join(path, name.replace('upper', 'lower')))[0].transpose()
        source_root = np.load(os.path.join(path, name.replace('upper', 'root')))[0].transpose()
        prefix = name.replace('_upper.npy', '')
        main(source_upper, source_lower, source_root, model, save_path, prefix, model_name='vqvae_ulr')



if __name__ == '__main__':
    '''
    cd codebook/
    python VisualizeCodebook.py --config=./configs/codebook.yml --train --gpu 0
    '''

    model = VQVAE_ulr(config.VQVAE, 7 * 16)  # n_joints * n_chanels
    model_path = "./result/my_codebook/codebook_checkpoint_best.bin"
    model = nn.DataParallel(model, device_ids=[eval(i) for i in config.no_cuda])
    model = model.to(mydevice)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_dict'])
    model = model.eval()

    # # test orignal latent
    # source_upper = np.load('../retargeting/datasets/bvh2upper_lower_root/Trinity/Recording_006_upper.npy')[0].transpose()
    # # print(source_upper.shape)       # (3164, 64)
    # save_path = "./result/inference/Trinity"
    # prefix = 'train_codebook_bvh2upper_lower_root_again'
    # main_2(source_upper, model, save_path, prefix)


    # generate code from latent
    ZEGGS_path = '../retargeting/datasets/Trinity_ZEGGS/bvh2upper_lower_root/ZEGGS/'
    Trinity_path = '../retargeting/datasets/Trinity_ZEGGS/bvh2upper_lower_root/Trinity/'
    ZEGGS_save_path = '../retargeting/datasets/Trinity_ZEGGS/VQVAE_result/ZEGGS'
    Trinity_save_path = '../retargeting/datasets/Trinity_ZEGGS/VQVAE_result/Trinity/'
    fold2code(Trinity_path, Trinity_save_path)
    fold2code(ZEGGS_path, ZEGGS_save_path)



    # test generate code
    # test_file = '../result/inference/Recording_006.npy'
    # path = os.path.split(test_file)[0]
    # name = os.path.split(test_file)[1]
    # code_upper = np.load(os.path.join(path, 'code_upper_' + name))
    # code_lower = np.load(os.path.join(path, 'code_lower_' + name))
    # code_root = np.load(os.path.join(path, 'code_root_' + name))
    # save_path = "../result/inference/Trinity/"

    # visualize_code(model, save_path, name, code_upper, code_lower=None, code_root=None, model_name='vqvae_ulr_2')
