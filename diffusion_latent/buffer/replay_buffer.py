import numpy as np
import torch
from collections import namedtuple
from loguru import logger


RolloutBufferSamples = namedtuple('RolloutBufferSamples', [
    'musics',
    'poses',
    'returns',
    'logprobs',
    'times',
])


class RolloutBuffer:
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.
    """

    def __init__(self, config) -> None:

        # pointer
        self.pos = 0
        self.full = False
        self.returns_ready = False

        # device
        self.device = torch.device(config.device)
        logger.info(f"buffer device is {self.device}")

        # buffer shape
        self.buffer_size = config.buffer_size   # for ppo, buffer_size should be the interaction steps
        self.traj_len = config.traj_len
        self.music_feat_dim = config.music_feat_dim # music_feat_dim

        # return
        self.gamma = config.gamma
        self.reset()
    
    def reset(self):
        """
        reset buffer and pointer
        """
        # pointer
        self.ptr = 0
        self.full = False
        self.returns_ready = False

        # buffer
        # obs traj_len = 16
        self.musics = np.zeros((self.buffer_size, self.traj_len, self.music_feat_dim), dtype=np.float32)
        self.poses = np.zeros((self.buffer_size, self.traj_len), dtype=np.int64)
        # reward
        self.rewards = np.zeros((self.buffer_size, self.traj_len), dtype=np.float32) # pure reward (without kl penalty)
        # critic target
        self.returns = np.zeros((self.buffer_size, self.traj_len), dtype=np.float32) # advantages + values
        # actor output
        self.logprobs = np.zeros((self.buffer_size, self.traj_len), dtype=np.float32)
        # t output
        self.times = np.zeros((self.buffer_size, 1), dtype=np.int64)

        logger.info("buffer reset complete!")

    def compute_returns(self):
        """
        compute return and advantage
        """
        # assert self.full, "For On-policy Algorithm, rollout buffer should be full after once rollout!"
        for step in reversed(range(self.traj_len)):
            if step == self.traj_len - 1:
                self.returns[:self.ptr, step] = self.rewards[:self.ptr, step]
            else:
                self.returns[:self.ptr, step] = self.rewards[:self.ptr, step] + self.gamma * self.returns[:self.ptr, step+1]

        self.returns_ready = True

        logger.info("buffer return calc complete. adv ready!")

    def add(
        self,
        music, # [B, T, music_feat_dim]
        pose, # [B, T]
        reward, # [B, T]
        logprob, # [B, T]
        time, #[B, 1]
        ) -> None:
        """
        add new experiences
        """
        B, T = reward.shape
        assert T == self.traj_len, f"the length of asserted data should be {self.traj_len}, rather than {T}"
        
        # capacity check
        assert self.ptr + B <= self.buffer_size

        # shape check
        assert music.shape == (B, T, self.music_feat_dim)
        assert pose.shape == (B, T)
        assert reward.shape == (B, T)
        assert logprob.shape == (B, T)
        assert time.shape == (B, 1)

        # add
        self.musics[self.ptr:self.ptr+B] = music
        self.poses[self.ptr:self.ptr+B] = pose
        self.rewards[self.ptr:self.ptr+B] = reward
        self.logprobs[self.ptr:self.ptr+B] = logprob
        self.times[self.ptr: self.ptr+B] = time

        self.ptr += B
        # logger.info(f"{B} items of data add complete!")

        # if self.ptr == self.buffer_size:
        #     self.full = True
        #     logger.info("buffer is full and rollout complete!")
        

    def get(self, batch_size):
        """
        return NamedTuple
        """
        # assert self.full, "For On-policy Algorithm, rollout buffer should be full before getting rollouts!"
        assert self.returns_ready, "You should calc adv before getting rollouts"

        indices = np.random.permutation(self.ptr)
        for start_idx in range(0, self.ptr, batch_size):
            end_idx = start_idx + batch_size if start_idx + batch_size <= self.ptr else self.ptr
            yield self._get_samples(indices[start_idx: end_idx])

    def _get_samples(self, batch_inds):
        data = (
            self.musics[batch_inds],
            self.poses[batch_inds],
            self.returns[batch_inds],
            self.logprobs[batch_inds],
            self.times[batch_inds],
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)
    
    @property
    def capacity(self):
        return self.buffer_size
    
    @property
    def current_size(self):
        return self.ptr
    
    @property
    def is_full(self):
        return self.full
    
    @property
    def is_returns_ready(self):
        return self.returns_ready
    
    @property
    def summary(self):
        return {
            'reward_mean': self.rewards[:self.ptr].sum(axis=-1).mean(),
            'reward_std': self.rewards[:self.ptr].sum(axis=-1).std(),
        }


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from easydict import EasyDict

    # get config from yaml file
    with open("/ceph/home/wangzl21/Projects/My_3/deep-motion-editing/diffusion_latent/configs/reinforce.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    pprint(config)
    config = EasyDict(config)

    # initialize buffer
    buffer = RolloutBuffer(config=config.buffer)
    print(buffer.capacity, buffer.current_size, buffer.is_full, buffer.is_returns_ready)

    # fake data
    # B = config.buffer.buffer_size // 2
    B = 10
    T = config.buffer.traj_len
    add_data_1 = {
        "music": np.random.randn(B, T, config.buffer.music_feat_dim),
        "pose": np.random.randint(0, 512, size=(B, T)),
        "reward": np.random.randn(B, T),
        "logprob": np.random.randn(B, T),
        "time": np.random.randn(B, 1),
    }
    add_data_2 = {
        "music": np.random.randn(B, T, config.buffer.music_feat_dim),
        "pose": np.random.randint(0, 512, size=(B, T)),
        "reward": np.random.randn(B, T),
        "logprob": np.random.randn(B, T),
        "time": np.random.randn(B, 1),
    }
    buffer.add(**add_data_1)
    buffer.add(**add_data_2)
    buffer.compute_returns()

    print(buffer.capacity, buffer.current_size, buffer.is_full, buffer.is_returns_ready)

    # get
    for i, rollout_data in enumerate(buffer.get(B)):
        logger.info(f"{i+1}th batch:")
        print(
            # rollout_data.actions.shape,
            rollout_data.logprobs.shape,
            rollout_data.returns.device,
            rollout_data.poses.dtype
        )
    
    pprint(buffer.summary)

