"""
Patch Encoders / Decoders as used by DeepMind in their sonnet repo example:
https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
"""
import pdb

import torch
from torch import nn, einsum
import torch.nn.functional as F

# -----------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, input_channels, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, input_channels, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out += x
        out = F.relu(out)
        return out


class DeepMindEncoder(nn.Module):

    def __init__(self, input_channels):         # 7 * 16
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_channels, input_channels // 2),
            nn.ReLU(True),
        )

    def forward(self, x):       # [8, 30, 16 * 7] -> [8, 16, 30, 7]
        x = self.net(x)     # -> [8, 30, 128]
        x = x.unsqueeze(-1)     # -> [8, 30, 128, 1]
        x = x.permute(0, 2, 1, 3)    # -> [8, 128, 30, 1]
        return x


class DeepMindDecoder(nn.Module):

    def __init__(self, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(64, out_channels),
            nn.Tanh()
        )

    def forward(self, x):       # [batch, 64, 30, 1]
        x = x.permute(0, 2, 3, 1)   # -> [batch, 30, 1, 112]
        x = self.net(x)     #
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])  # [8, 30, 112]
        return x
