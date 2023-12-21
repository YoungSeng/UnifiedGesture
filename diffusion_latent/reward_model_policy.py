# reward model training and evaluation

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import socket
import wandb
import argparse
import yaml
from pprint import pprint
from easydict import EasyDict
from model import RewardTransformer
from tqdm import tqdm
from loguru import logger

from torch.utils.data import Dataset

class DemoDataset(Dataset):
    def __init__(self, observations, actions, labels) -> None:
        super().__init__()
        assert (len(observations) == len(actions) == len(labels)), "Dataset Length Error!"
        self.observations = observations
        self.actions = actions
        self.labels = labels

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int):

        return self.observations[index], self.actions[index], self.labels[index]


class RewardModelPolicy():

    def __init__(self, config) -> None:
        self.config = config

        # determinate convolution algorithm to save time
        torch.backends.cudnn.benchmark = True
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            logger.info("use device cuda:0")
        else:
            logger.error("cuda is indispensible!")
            raise
        torch.cuda.manual_seed(config.seed)

        self._build()

        # wandb
        wandb.init(
            config=self.config,
            **config.wandb
        )

    def train(self):
        self.reward_model.train()
        loss_criterion = nn.CrossEntropyLoss()
        timesteps = torch.arange(0, self.config.model.context_len).unsqueeze(0).repeat(self.config.batch_size, 1)
        timesteps = timesteps.to(self.device)
        for epoch_i in range(self.config.epoch):
            epoch_loss = []
            train_correct = 0
            train_total = 0
            for batch_i, batch in enumerate(tqdm(self.training_loader)):

                # set input
                observation, action, label = batch
                obs_i, obs_j = observation
                act_i, act_j = action
                obs_i, obs_j = obs_i.to(self.device), obs_j.to(self.device)
                act_i, act_j = act_i.to(self.device), act_j.to(self.device)
                label = label.to(self.device)   # (B,)

                # forward
                rewards_i = self.reward_model(timesteps, obs_i, act_i)
                rewards_j = self.reward_model(timesteps, obs_j, act_j)

                # calculate loss
                returns_i = torch.sum(rewards_i, dim=1) # (B, 1)
                returns_j = torch.sum(rewards_j, dim=1) # (B, 1)
                returns = torch.cat([returns_i, returns_j], 1)  # (B, 2)
                loss = loss_criterion(returns, label)

                # backward and optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.config.clip_grad_norm)
                self.optimizer.step()

                item_loss = loss.item()
                epoch_loss.append(item_loss)

                train_prediction = torch.argmax(returns, 1)
                train_correct += (train_prediction == label).sum().float()
                train_total += len(label)
                
            logger.info(f"epoch {epoch_i+1} loss {sum(epoch_loss)/len(epoch_loss)}")
            wandb.log({'train epoch loss':sum(epoch_loss)/len(epoch_loss)}, step=epoch_i+1)
            wandb.log({'train accuracy':(train_correct/train_total).cpu().detach().data.numpy()}, step=epoch_i+1)

            checkpoint = {
                'model': self.reward_model.state_dict(),
                'config': self.config,
                'epoch': epoch_i
            }
                
            # Save checkpoint
            if (epoch_i+1) % self.config.save_per_epochs == 0 and epoch_i!=0:
                filename = os.path.join(self.ckptdir, f'epoch_{epoch_i+1}.pt')
                torch.save(checkpoint, filename)

            if (epoch_i+1) % self.config.test_freq == 0:
                with torch.no_grad():
                    print("Evaluation...")
                    self.reward_model.eval()
                    correct = 0
                    total = 0
                    test_loss = 0.
                    for batch_i, batch in enumerate(tqdm(self.testing_loader, desc='evaluation')):

                        # set input
                        observation, action, label = batch
                        obs_i, obs_j = observation
                        act_i, act_j = action
                        obs_i, obs_j = obs_i.to(self.device), obs_j.to(self.device)
                        act_i, act_j = act_i.to(self.device), act_j.to(self.device)
                        label = label.to(self.device)   # (B,)

                        # forward
                        rewards_i = self.reward_model(timesteps, obs_i, act_i)
                        rewards_j = self.reward_model(timesteps, obs_j, act_j)

                        # calculate loss
                        returns_i = torch.sum(rewards_i, dim=1) # (B, 1)
                        returns_j = torch.sum(rewards_j, dim=1) # (B, 1)
                        returns = torch.cat([returns_i, returns_j], 1)  # (B, 2)

                        test_loss += loss_criterion(returns, label).item()
                        prediction = torch.argmax(returns, 1)
                        correct += (prediction == label).sum().float()
                        total += len(label)

                    logger.info('Evaluation complete! Accuracy: %f'%((correct/total).cpu().detach().data.numpy()))
                    wandb.log({'test epoch loss':test_loss/(batch_i+1)}, step=epoch_i+1)
                    wandb.log({'test accuracy':(correct/total).cpu().detach().data.numpy()}, step=epoch_i+1)

                self.reward_model.train()

        logger.info("finished training")

    def eval(self):
        pass

    def _build(self):
        self._dir_setting()
        self._build_model()
        self._build_train_loader()
        self._build_test_loader()
        self._build_optimizer()

    def _build_model(self):
        self.reward_model = RewardTransformer(self.config.model)

        self.reward_model.to(self.device)


    def _build_train_loader(self):
        """
        data: list
            noise_level0:
                trajs0:
                    {'state': s0 s1, 'action': a0 a1}
        
        """
        logger.info("build train loader")
        config = self.config.data
        context_len = self.config.model.context_len
        # load data
        with open(config.train_dir, 'rb') as f:
            pre_expert_data = pickle.load(f)
        n_noise_levels = len(pre_expert_data)
        assert n_noise_levels > 1, "noise level should be geq 1!"
        n_trajs_each_level = len(pre_expert_data[0])
        assert n_trajs_each_level >= 2, "trajs of each noise level should be geq 2!"

        num_snippets = config.n_snippets_train

        # collect training data
        self.training_observations, self.training_actions, self.training_labels= [], [], []
        for _ in range(num_snippets):
            level_i, level_j = np.random.choice(n_noise_levels, size=(2,), replace=False) # replace=False means no same level
            index_i, index_j = np.random.choice(n_trajs_each_level, size=(2,), replace=True)

            # plus 1 to remove start action
            obs_i = pre_expert_data[level_i][index_i]['state']
            act_i = pre_expert_data[level_i][index_i]['action']
            obs_j = pre_expert_data[level_j][index_j]['state']
            act_j = pre_expert_data[level_j][index_j]['action']

            label = int(level_i <= level_j)

            self.training_observations.append((obs_i, obs_j))
            self.training_actions.append((act_i, act_j))
            self.training_labels.append(label)

        self.training_loader = torch.utils.data.DataLoader(
            DemoDataset(self.training_observations, self.training_actions, self.training_labels),
            num_workers=8,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )


    def _build_test_loader(self):
        logger.info("build test loader")
        config = self.config.data
        context_len = self.config.model.context_len
        # load data
        with open(config.test_dir, 'rb') as f:
            pre_expert_data = pickle.load(f)
        n_noise_levels = len(pre_expert_data)
        assert n_noise_levels > 1, "noise level should be geq 1!"
        n_trajs_each_level = len(pre_expert_data[0])
        assert n_trajs_each_level >= 2, "trajs of each noise level should be geq 2!"

        num_snippets = config.n_snippets_test

        # collect training data
        self.testing_observations, self.testing_actions, self.testing_labels= [], [], []
        for _ in range(num_snippets):
            level_i, level_j = np.random.choice(n_noise_levels, size=(2,), replace=False) # replace=False means no same level
            index_i, index_j = np.random.choice(n_trajs_each_level, size=(2,), replace=True)

            # plus 1 to remove start action
            obs_i = pre_expert_data[level_i][index_i]['state']
            act_i = pre_expert_data[level_i][index_i]['action']
            obs_j = pre_expert_data[level_j][index_j]['state']
            act_j = pre_expert_data[level_j][index_j]['action']

            label = int(level_i <= level_j)

            self.testing_observations.append((obs_i, obs_j))
            self.testing_actions.append((act_i, act_j))
            self.testing_labels.append(label)

        self.testing_loader = torch.utils.data.DataLoader(
            DemoDataset(self.testing_observations, self.testing_actions, self.testing_labels),
            num_workers=8,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )

    def _build_optimizer(self):
        logger.info("build optimizer")
        config = self.config.optimizer
        try:
            optimizer = getattr(torch.optim, config.type)
        except Exception:
            raise NotImplementedError("Not implemented optimizer method " + config.type)
        self.optimizer = optimizer(self.reward_model.parameters(), **config.kwargs)

    def _dir_setting(self):
        logger.info("set dir")
        self.expname = self.config.expname
        self.experiment_dir = os.path.join("./", "experiments")
        self.expdir = os.path.join(self.experiment_dir, self.expname)

        if not os.path.exists(self.expdir):
            os.mkdir(self.expdir)

        self.ckptdir = os.path.join(self.expdir, "ckpt")
        if not os.path.exists(self.ckptdir):
            os.mkdir(self.ckptdir)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Reward Model for Music2Pose')
    parser.add_argument('--config', default='configs/reward_model.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    config = EasyDict(config)

    agent = RewardModelPolicy(config)
    print(config)

    agent.train()