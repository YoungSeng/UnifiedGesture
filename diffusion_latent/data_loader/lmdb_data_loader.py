# ! pip install configargparse easydict

import logging
import os
import pdb

import numpy as np
import lmdb as lmdb
import torch
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

import sys

[sys.path.append(i) for i in ['.', '..']]

from data_loader.data_preprocessor import DataPreprocessor
import pyarrow


class TrinityDataset(Dataset):
    def __init__(self, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, model=None):
        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.lang_model = None

        logging.info("Reading data '{}'...".format(lmdb_dir))
        if model is not None:
            if 'WavLM' in model:
                preloaded_dir = lmdb_dir + '_cache_' + model
        else:
            preloaded_dir = lmdb_dir + '_cache'
        if not os.path.exists(preloaded_dir):
            data_sampler = DataPreprocessor(lmdb_dir, preloaded_dir, n_poses,
                                            subdivision_stride, pose_resampling_fps)
            data_sampler.run()
        else:
            logging.info('Found pre-loaded samples from {}'.format(preloaded_dir))

        # init lmdb
        # map_size = 1024 * 20  # in MB
        # map_size <<= 20  # in B
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)  # default 10485760
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)

            sample = pyarrow.deserialize(sample)
            # poses, code, style, wavlm, aux_info = sample
            poses, code, style, wavlm = sample

        # to tensors
        pose_seq = torch.from_numpy(poses).float()
        # vel_seq = torch.from_numpy(vel).float()
        # acc_seq = torch.from_numpy(acc).float()
        # print('pose_seq.shape', pose_seq.shape)
        # pose_seq = torch.cat([pose_seq, vel_seq, acc_seq], dim=-1)
        code = torch.from_numpy(code).float()
        style = torch.from_numpy(style).float()
        wavlm = torch.from_numpy(wavlm).float()

        return pose_seq, code, style, wavlm


if __name__ == '__main__':
    '''
    cd ./diffusion_latent/
    python data_loader/lmdb_data_loader.py --config=./configs/all_data.yml --train --no_cuda 3 --gpu 3
    '''

    from configs.parse_args import parse_args
    import os
    import yaml
    from pprint import pprint
    from easydict import EasyDict
    from torch.utils.data import DataLoader

    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    args = EasyDict(config)


    val_dataset = TrinityDataset(args.val_data_path,
                                       n_poses=args.n_poses,
                                       subdivision_stride=args.subdivision_stride,
                                       pose_resampling_fps=args.motion_resampling_framerate, model='WavLM_36_aux')


    valid_loader = DataLoader(dataset=val_dataset, batch_size=128,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True)

    print(len(valid_loader))
    for batch_i, batch in enumerate(valid_loader, 0):
        pose_seq, code, style, wavlm = batch     # [128, 88, 1141], -,  [128, 6], [128, 70400], [128, 88, 13]
        # [128, 32, 112], [128, 16], [128, 7], [128, 32, 1024]
        print(batch_i)
        pdb.set_trace()
        break

    train_dataset = TrinityDataset(args.train_data_path,
                                   n_poses=args.n_poses,
                                   subdivision_stride=args.subdivision_stride,
                                   pose_resampling_fps=args.motion_resampling_framerate, model='WavLM_36_aux')

    train_loader = DataLoader(dataset=train_dataset, batch_size=128,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True)