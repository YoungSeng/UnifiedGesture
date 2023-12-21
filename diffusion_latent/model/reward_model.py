import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml
from pprint import pprint
from easydict import EasyDict


class Attention(nn.Module):
    """
    Vannilla multi-head self-attention layer without last projection
    """
    def __init__(self, config):
        super().__init__()
        assert config.hid_dim % config.n_heads == 0, "config.hid_dim % config.n_heads != 0!"
        # key, query, value projections for all heads
        self.key = nn.Linear(config.hid_dim, config.hid_dim)
        self.query = nn.Linear(config.hid_dim, config.hid_dim)
        self.value = nn.Linear(config.hid_dim, config.hid_dim)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # head
        self.n_heads = config.n_heads
        # output projection
        self.proj = nn.Linear(config.hid_dim, config.hid_dim)
        # mask
        if config.causal:
            self.register_buffer("mask", torch.tril(torch.ones(config.context_len*2, config.context_len*2)).view(1, 1, config.context_len*2, config.context_len*2))

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads (B, n_head, T, hid_dim//n_head)
        k = self.key(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if hasattr(self, "mask"):
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.hid_dim)
        self.ln2 = nn.LayerNorm(config.hid_dim)
        self.attn = Attention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.hid_dim, 4 * config.hid_dim),
            nn.GELU(),
            nn.Linear(4 * config.hid_dim, config.hid_dim),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        if self.config.norm_first:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        else:
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.mlp(x))
        return x


class RewardTransformer(nn.Module):
    """
    Reward Transformer for reward prediction
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # embedding
        # self.embed_ln = nn.LayerNorm(config.hid_dim)
        self.timestep_embed = nn.Embedding(config.context_len, config.hid_dim)
        self.pose_embed = nn.Embedding(config.n_action, config.hid_dim)
        # self.action_embed = nn.Linear(2, config.hid_dim)
        self.music_embed = nn.Linear(config.n_music, config.hid_dim)
        # transformer
        blocks = nn.ModuleList([Block(config) for _ in range(config.n_blocks)])
        self.transformer = nn.Sequential(*blocks)
        # reward prediction head
        self.predict_reward = nn.Linear(2*config.hid_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # module.weight.data.uniform_(math.sqrt(6.0/sum(module.weight.size())))
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, timesteps, music, pose):
        """
        Predict the reward with music and pose by transformer
        """
        B, T = timesteps.size()
        assert T == self.config.context_len, "T is not equal with text_len"

        # embedding
        timestep_embeddings = self.timestep_embed(timesteps)
        music_embeddings = self.music_embed(music) + timestep_embeddings
        pose_embeddings = self.pose_embed(pose) + timestep_embeddings

        # stack into input
        h = torch.stack((music_embeddings, pose_embeddings), dim=1).permute(0, 2, 1, 3).reshape(B, 2*T, self.config.hid_dim)
        # h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)
        h = h.reshape(B, T, 2*self.config.hid_dim)

        rewards = self.predict_reward(h)

        return rewards

        

if __name__ == "__main__":
    """
    Reward Model python test
    """
    # load config
    config_file_path = "/ceph/home/wangzl21/Projects/My_3/deep-motion-editing/diffusion_latent/configs/reward_model.yaml"
    with open(config_file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    pprint(config)
    config = EasyDict(config)

    # model initialization
    reward_model = RewardTransformer(config=config.model)
    print(reward_model)

    # other configs
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        raise
    reward_model.to(device)

    # fake data manufication
    B, T = 64, config.model.context_len
    timesteps = torch.arange(0, T, 1).unsqueeze(0).repeat(B, 1).cuda()
    music = torch.randn((B, T, config.model.n_music)).cuda()
    pose = torch.randint(0, 512, (B, T)).cuda()

    # forward
    rewards = reward_model(timesteps, music, pose)

    print(rewards.shape)    # (B, T, 1)





    
