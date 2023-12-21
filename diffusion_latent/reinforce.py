import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
import yaml
import sys
import subprocess
import pdb
import librosa
import math
import glob

from sample_rl import single_inference, main, args
from tqdm import tqdm
from loguru import logger
from buffer.replay_buffer import RolloutBuffer
from environment.environment import Arena
from easydict import EasyDict
from datetime import datetime
from pprint import pprint
from torch.distributions.categorical import Categorical

# from utils_.model_util import create_gaussian_diffusion, load_model_wo_clip
# [sys.path.append(i) for i in ['.', '..', '../model', '../codebook']]
# from model.mdm import MDM
# from codebook.models.vqvae import VQVAE_ulr
# from WavLM import WavLM, WavLMConfig


class REINFORCE:

    def __init__(self, config) -> None:
        self.config = config

        # determinate convolution algorithm to save time
        torch.backends.cudnn.benchmark = True
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            raise RuntimeError("GPU is not available!")
        torch.cuda.manual_seed(config.seed)

        self._build()

        # save config and commit id
        with open(f'{self.expdir}/config.yaml', 'w+') as log_file:
            save_config = easydict_to_dict(self.config)
            # save_config['commit_id'] = get_git_commit_id()
            yaml.dump(save_config, log_file, default_flow_style=False)

    # def train(self):
    #     config = self.config.policy.train
        
    #     for epoch in range(1, config.num_epochs+1):
    #         logger.info(f"Epoch {epoch} start!")

    #         if self.buffer.is_full:
    #             logger.info("buffer full, reset!")
    #             self.buffer.reset()
            
    #         # get rollouts
    #         logger.info("Getting rollout")
    #         self.get_rollout()
    #         logger.info("Rollout finish, now training start!")

    #         buffer_data = self.buffer.summary
    #         logger.info(f"Epoch {epoch}: reward_mean {buffer_data['reward_mean']}, reward_std {buffer_data['reward_std']}")

    #         # training
    #         # get generator
    #         train_batches = self.buffer.get(config.train_batch_size)
    #         for batch_index, buf_data in enumerate(train_batches):
    #             loss = self.calc_loss(self, buf_data)
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.vqvae_model.module.parameters(), config.max_grad_norm)
    #             torch.nn.utils.clip_grad_norm_(self.diffusion_model.module.parameters(), config.max_grad_norm)
    #             self.optimizer.step()

    #         logger.info(f"In epoch {epoch}, batch {batch_index+1}: loss: {loss.item()}")

    #     logger.info(f"Epoch {epoch} training finished!")

    # def get_rollout(self):
    #     config = self.config
    #     ZEGGS_audio_fold = '../dataset/ZEGGS/all_speech/'
    #     Trinity_audio_fold = '../dataset/Trinity/all_speech/'

    #     train_audio_list = sorted(glob.glob(ZEGGS_audio_fold + '/*.npy') + glob.glob(Trinity_audio_fold + '/*.npy'))

    #     for audio in train_audio_list:
    #         name = os.path.split(audio)[1][:-4]
    #         audio_wav = np.load(audio)[0]
    #         logger.info(name)
    #         _, code, wavlm_feat, distance, t = sample.single_inference(audio, config, self.sample_fn, self.wavlm_model, self.vqvae_model, self.diffusion_model, audio_wav=audio_wav, time_step=True, return_seed=True, w_grad=False)

    #         # motion_list.append(motion)
    #         code = code.squeeze(1)
    #         wavlm_feat = wavlm_feat.squeeze(1)[:code.shape[0]]

    #         code = code.reshape(-1, config.n_poses//2)
    #         distance = distance.reshape(-1, config.n_poses//2, 512)
    #         wavlm_feat = wavlm_feat.reshape(-1, config.n_poses//2, 1024)

    #         # cal reward
    #         rewards = self.env.get_rewards(code, wavlm_feat).cpu().numpy() # [B, 18]

    #         pi = Categorical(logits=-distance)

    #         # calc logprobs
    #         logprobs = pi.log_prob(code)

    #         # add data to buffer
    #         rollouts = {
    #             'music': wavlm_feat,
    #             'pose': code,
    #             'reward': rewards.cpu().numpy(),
    #             'logprob': logprobs.cpu().numpy(),
    #             'time': t,
    #         }

    #         self.buffer.add(**rollouts)

    #     self.buffer.compute_returns()

    def train(self):
        config = self.config.policy.train

        with open('configs/all_data.yml') as f:
            all_data_config = yaml.safe_load(f)
        all_data_config = EasyDict(all_data_config)
        all_data_config.train = True
        all_data_config.gpu = 0
        all_data_config.no_cuda = ['0']
        all_data_config.overwrite = False
        all_data_config.config = './configs/codebook.yml'
        
        for epoch in range(1, config.num_epochs+1):
            ZEGGS_audio_fold = '../dataset/ZEGGS/all_speech/'
            Trinity_audio_fold = '../dataset/Trinity/all_speech/'

            train_audio_list = sorted(glob.glob(ZEGGS_audio_fold + '/*.npy') + glob.glob(Trinity_audio_fold + '/*.npy'))[:1]

            codes_list = []
            returns_list = []
            audio_list = []
            audio_wav_list = []
            t_list = []

            for audio_index, audio in enumerate(train_audio_list):
                name = os.path.split(audio)[1][:-4]
                audio_wav = np.load(audio)[0]
                audio_list.append(audio)
                audio_wav_list.append(audio_wav)
                _, code, wavlm_feat, distance, t = single_inference(audio, all_data_config, self.sample_fn, self.wavlm_model, self.vqvae_model, self.diffusion_model, audio_wav=audio_wav, time_step=True, return_seed=True, w_grad=False)
                logger.info(audio)
                logger.info(code[:20])
                logger.info(distance[0,-20:])
                t_list.append(t.squeeze(-1))

                # motion_list.append(motion)
                code = code.squeeze(1)
                wavlm_feat = wavlm_feat.squeeze(1)[:code.shape[0]]

                code = code.reshape(-1, all_data_config.n_poses//2)
                distance = distance.reshape(-1, all_data_config.n_poses//2, 512)
                wavlm_feat = wavlm_feat.reshape(-1, all_data_config.n_poses//2, 1024)

                # cal reward
                with torch.no_grad():
                    rewards = self.env.get_rewards(torch.tensor(code, device=self.device), torch.tensor(wavlm_feat, device=self.device)) # [B, 18]
                returns = torch.zeros_like(rewards)

                for step in reversed(range(config.traj_len)):
                    if step == config.traj_len - 1:
                        returns[:, step] = rewards[:, step]
                    else:
                        returns[:, step] = rewards[:, step] + config.gamma * returns[:, step+1]
                # mu, sigma = returns.mean(), returns.std()
                # returns = (returns-mu)/sigma

                pi = Categorical(logits=-torch.tensor(distance, device=self.device))  

                # calc logprobs
                my_code = pi.sample()   # [B, 18]
                # logprobs = pi.log_prob(my_code)    # [B, 18]

                # logprobs_list.append(logprobs)
                codes_list.append(my_code)
                returns_list.append(returns)

                # loss = -(logprobs * returns).mean()


            for batch_index, (batch_audio, batch_audio_wav, batch_code, batch_returns, times) in enumerate(zip(audio_list, audio_wav_list, codes_list, returns_list, t_list)):
                _, code, wavlm_feat, distance, t = single_inference(batch_audio, all_data_config, self.sample_fn, self.wavlm_model, self.vqvae_model, self.diffusion_model, audio_wav=batch_audio_wav, time_step=True, time=times, return_seed=True, w_grad=True)
                distance = distance.reshape(-1, all_data_config.n_poses//2, 512)
                # pdb.set_trace()
                pi = Categorical(logits=-distance)
                logprobs = pi.log_prob(batch_code)
                pdb.set_trace()
                loss = (-logprobs * batch_returns).mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.vqvae_model.module.parameters(), config.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), config.max_grad_norm)
                self.optimizer.step()
                logger.info(f"In epoch {epoch}, batch_idx: {batch_index}: loss: {loss.item()}, logprobs: {logprobs[0, :5]}, returns: {batch_returns[0, :5]}")
                # pdb.set_trace()


        logger.info(f"Epoch {epoch} training finished!")

        if epoch % config.save_epoch_interval == 0:
            vqvae_model_path = os.path.join(self.ckptdir, f'vqvae_epoch_{epoch}.bin')
            vqvae_checkpoint = {
                'model_dict': self.vqvae_model.state_dict(),
                'config': self.config,
                'epoch': epoch
            }
            torch.save(vqvae_checkpoint, vqvae_model_path)
            import blobfile as bf
            with bf.BlobFile(bf.join(self.ckptdir, f'diffusion_epoch_{epoch}.pt'), "wb") as f:
                torch.save(self.diffusion_model.state_dict(), f)


    def _build(self):
        self._dir_setting()
        self._build_model()
        self._build_optimizer()
        # self._build_buffer()
        self._build_env()

    def _dir_setting(self):
        self.expname = self.config.expname
        self.experiment_dir = os.path.join("./", "experiments")
        self.expdir = os.path.join(self.experiment_dir, self.expname)

        if not os.path.exists(self.expdir):
            os.mkdir(self.expdir)

        self.ckptdir = os.path.join(self.expdir, "ckpt")
        if not os.path.exists(self.ckptdir):
            os.mkdir(self.ckptdir)

    def _build_model(self):
        save_dir = './result/Trinity'
        model_path = './result/my_diffusion/model000000000.pt'
        self.config.diff_model_path = model_path
        # model_path = '/ceph/home/wangzl21/Projects/My_3/deep-motion-editing/diffusion_latent/experiments/reinforce_diffusion_baseline_onlydiff_seed0/ckpt/diffusion_epoch_1.pt'
        self.sample_fn, self.wavlm_model, self.vqvae_model, self.diffusion_model = main(save_dir, model_path)
        for param in self.vqvae_model.module.parameters():    # TODO
            param.requires_grad=False

        # # vqvae init
        # vqvae_model = VQVAE_ulr(config.VQVAE, 7 * 16)  # n_joints * n_chanels
        # vqvae_model_path = "../codebook/result/train_codebook_upper_lower_root_downsample2/codebook_checkpoint_best.bin"
        # vqvae_model = nn.DataParallel(vqvae_model, device_ids=self.device)
        # vqvae_model = vqvae_model.to(self.device)
        # vqvae_checkpoint = torch.load(vqvae_model_path, map_location=torch.device('cpu'))
        # vqvae_model.load_state_dict(vqvae_checkpoint['model_dict'])
        # self.vqvae_model = vqvae_model.eval()

        # # wavlm init
        # wavlm_model_path = './wavlm_cache/WavLM-Large.pt'
        # # load the pre-trained checkpoints
        # wavlm_checkpoint = torch.load(wavlm_model_path, map_location=torch.device('cpu'))
        # wavlm_cfg = WavLMConfig(wavlm_checkpoint['cfg'])
        # wavlm_model = WavLM(wavlm_cfg)
        # wavlm_model = wavlm_model.to(self.device)
        # wavlm_model.load_state_dict(wavlm_checkpoint['model'])
        # self.wavlm_model = wavlm_model

        # # diffusion init
        # diff_model_path = './result/inference/new_DiffuseStyleGesture_model3_256/model001100000.pt'
        # diff_model = MDM(modeltype='', njoints=16 * 7 * 3, nfeats=1, translation=True, pose_rep='rot6d', glob=True,
        #             glob_rot=True, cond_mode='cross_local_attention3_style1', action_emb='tensor', audio_feat='wavlm',
        #             arch='trans_enc', latent_dim=256, n_seed=4, cond_mask_prob=0.1)
        # diffusion = create_gaussian_diffusion()
        # diff_state_dict = torch.load(diff_model_path, map_location='cpu')
        # load_model_wo_clip(diff_model, diff_state_dict)
        # diff_model.to(self.device)
        # sample_fn = diffusion.p_sample_loop
        # self.sample_fn = sample_fn
        # self.diffusion_model = diff_model

    def _build_optimizer(self):
        config = self.config.policy.optimizer
        try:
            optimizer = getattr(torch.optim, config.type)
        except Exception:
            raise NotImplementedError("Not implemented optimizer method " + config.type)        
        # self.optimizer = optimizer(self.diffusion_model.parameters(), **config.kwargs)
        self.optimizer = optimizer(list(self.vqvae_model.module.parameters()) + list(self.diffusion_model.parameters()), **config.kwargs)

    # def _build_buffer(self):
    #     config = self.config.buffer
    #     self.buffer = RolloutBuffer(config)

    def _build_env(self):
        config = self.config.environment
        self.env = Arena(config)


def get_git_commit_id():
    """
    Retrieve the commit version of the current Git repository.
    Before running the code, please ensure that both the staging area and the working directory are clean, except for the config file.
    """
    # 调用 Git 命令获取当前提交版本的 ID
    commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    return commit_id

def easydict_to_dict(easy_dict):
    if isinstance(easy_dict, EasyDict):
        return dict((k, easydict_to_dict(v)) for k, v in easy_dict.items())
    elif isinstance(easy_dict, dict):
        return dict((k, easydict_to_dict(v)) for k, v in easy_dict.items())
    else:
        return easy_dict


if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=5 python reinforce.py --config=./configs/all_data.yml --no_cuda 0 --gpu 0
    """

    with open('configs/reinforce.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = EasyDict(config)
    config.expname = config.expname+f'_seed{config.seed}'

    logger.add(f'experiments/{config.expname}/{config.expname}.log')

    agent = REINFORCE(config)

    agent.train()
