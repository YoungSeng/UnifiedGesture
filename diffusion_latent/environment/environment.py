import torch
import torch.nn as nn
import numpy as np
import os
import json
import pickle
import pyarrow
import lmdb
import torch.nn.functional as F
from loguru import logger
from functools import partial
from model import RewardTransformer
from torch.utils.data import DataLoader, Dataset
from data_loader.data_preprocessor import DataPreprocessor


class Arena:

    def __init__(self, config):
        """
        lazy init
        """
        self.config = config
        self._build()

    def reset(self):
        try:
            data = next(self.train_data_generator)
        except StopIteration:
            self.train_data_generator = iter(self.train_data_loader)
            data = next(self.train_data_generator)
        return data

    def _build(self):
        self._build_reward_model()
        # self._build_train_loader()
        # self._build_test_loader()

    def _build_reward_model(self):
        # init reward model
        reward_model = RewardTransformer(self.config.reward_model)
        self.reward_model = reward_model.cuda()
        self.reward_model.eval()
        reward_model_ckpt = torch.load(self.config.reward_model.ckpt_path)
        self.reward_model.load_state_dict(reward_model_ckpt['model'])

    # def _build_train_loader(self):
    #     logger.info("build train data loader")
    #     data_config = self.config.data

    #     train_dataset = EnvTrinityDataset(data_config.train_data_path,
    #                                 n_poses=data_config.n_poses,
    #                                 subdivision_stride=data_config.subdivision_stride,
    #                                 pose_resampling_fps=data_config.motion_resampling_framerate, model='WavLM_36_aux')  # 62802

    #     train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=data_config.train_batch_size,
    #                             shuffle=data_config.train_shuffle, drop_last=True, num_workers=data_config.loader_workers, pin_memory=True)   # 200+
    #     self.train_data_loader = train_loader

    # def _build_test_loader(self):
    #     logger.info("build test data loader")
    #     data_config = self.config.data

    #     test_dataset = EnvTrinityDataset(data_config.test_data_path,
    #                                 n_poses=data_config.n_poses,
    #                                 subdivision_stride=data_config.subdivision_stride,
    #                                 pose_resampling_fps=data_config.motion_resampling_framerate, model='WavLM_36_aux')


    #     test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=data_config.test_batch_size,
    #                           shuffle=data_config.test_shuffle, drop_last=True, num_workers=data_config.loader_workers, pin_memory=True)
    #     self.test_data_loader = test_loader

    # def get_train_loader(self):
    #     return self.train_data_loader
    
    # def get_test_loader(self):
    #     return self.test_data_loader
    
    # def get_test_dance_names(self):
    #     return self.test_dance_names
    
    def get_rewards(self, idxs, cond):
        r"""Get the rewards under actions ``idxs`` and musics ``cond`` r_t = R(s_t,a_t)

        Note:
            The length of idxs and cond is longer than the context_len of network.
            We need to multiple feed forward refering to `self.sample()` function

        Args:
            idxs ((list[Tensor], list[Tensor])): two-tuple of list of tensors, representing the actions.
            cond (Tensor): Music tensor in ``B x 30 x n_musics``.

        Returns:
            rewards (Tensor of size ``B x 29``): rewards for given states and actions
        """
        context_len = self.config.reward_model.context_len
        device = cond.device
        B, T, _ = cond.shape    # [B, 30, n_music]

        assert T >= context_len, "the frame of music should be greater than context length"

        # set timesteps TODO 这里要不要转成 register_buffer？ 不需要
        timesteps = torch.arange(0, self.config.reward_model.context_len).unsqueeze(0).repeat(B, 1).to(device)

        # TODO 重新训练reward model，不需要错位music和action，而是直接丢弃第一帧的action
        rewards = self.reward_model(timesteps, cond, idxs)[:, :, 0]  # [B, 18]
 
        # for k in range(2, T-context_len+1):
        #     reward = self.reward_model(timesteps, cond[:, k:context_len+k, :], (x_up[:, k:context_len+k], x_down[:, k:context_len+k]))[:, -1:, 0]   # [B, 1]
        #     rewards = torch.cat([rewards, reward], dim=1) 

        if self.config.reward_model.sparse_reward:
            sparse_rewards = torch.zeros_like(rewards)
            sparse_rewards[:, -1] = rewards.sum(-1)
            rewards = sparse_rewards

        return rewards


class EnvTrinityDataset(Dataset):
    def __init__(self, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, model=None):
        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.lang_model = None

        logger.info("Reading data '{}'...".format(lmdb_dir))
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
            logger.info('Found pre-loaded samples from {}'.format(preloaded_dir))

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
        wavlm = F.interpolate(wavlm.unsqueeze(0).transpose(1, 2), size=18, align_corners=True, mode='linear').transpose(1, 2)[0]

        return pose_seq, code, style, wavlm
        

if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from easydict import EasyDict

    # get config from yaml file
    with open("configs/reinforce.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    pprint(config)
    config = EasyDict(config)

    env = Arena(config.environment)
    train_loader = env.get_train_loader()
    test_loader = env.get_test_loader()
    for _, ori_code, _, ori_wavlm in train_loader:
        print(ori_code.shape)
        print(ori_wavlm.shape)
        break
    for _, ori_code, _, ori_wavlm in test_loader:
        print(ori_code.shape)
        print(ori_wavlm.shape)
        break
