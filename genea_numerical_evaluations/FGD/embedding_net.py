import pdb

import torch
import torch.nn as nn


def ConvNormRelu(in_channels, out_channels, downsample=False, padding=0, batchnorm=True):
    if not downsample:
        k = 3
        s = 1
    else:
        k = 4
        s = 2

    conv_block = nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
    norm_block = nn.BatchNorm1d(out_channels)

    if batchnorm:
        net = nn.Sequential(
            conv_block,
            norm_block,
            nn.LeakyReLU(0.2, True)
        )
    else:
        net = nn.Sequential(
            conv_block,
            nn.LeakyReLU(0.2, True)
        )

    return net


class PoseEncoderConv(nn.Module):
    def __init__(self, dim, length):
        super().__init__()

        self.net = nn.Sequential(
            ConvNormRelu(dim, 32, batchnorm=True),
            ConvNormRelu(32, 64, batchnorm=True),
            ConvNormRelu(64, 64, True, batchnorm=True),
            nn.Conv1d(64, 32, 3)
        )


        if length == 120:
            in_channels = 1760
        else:
            assert False

        self.out_net = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(True),
            nn.Linear(128, 32),
        )

    def forward(self, poses):
        poses = poses.transpose(1, 2)  # to (bs, dim, seq)
        out = self.net(poses)
        out = out.flatten(1)
        z = self.out_net(out)

        return z


class PoseDecoderConv(nn.Module):
    def __init__(self, dim, length):
        super().__init__()

        if length == 30:
            out_channels = 120
        elif length == 60:
            out_channels = 240
        elif length == 90:
            out_channels = 360
        elif length == 120:
            out_channels = 120*4
        else:
            assert False

        self.pre_net = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(True),
            nn.Linear(64, out_channels),
        )

        self.net = nn.Sequential(
            nn.ConvTranspose1d(4, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(32, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(32, 32, 3),
            nn.Conv1d(32, dim, 3),
        )

    def forward(self, feat):
        out = self.pre_net(feat)
        out = out.view(feat.shape[0], 4, -1)
        out = self.net(out)
        out = out.transpose(1, 2)
        return out


class EmbeddingNet(nn.Module):
    def __init__(self, pose_dim, n_frames):
        super().__init__()
        self.pose_encoder = PoseEncoderConv(pose_dim, n_frames)
        self.decoder = PoseDecoderConv(pose_dim, n_frames)

    def forward(self, poses):
        poses_feat = self.pose_encoder(poses)
        out_poses = self.decoder(poses_feat)
        return poses_feat, out_poses


if __name__ == '__main__':  # model test
    '''
    python embedding_net.py
    '''
    n_frames = 120
    pose_dim = 33*3
    encoder = PoseEncoderConv(pose_dim, n_frames)
    decoder = PoseDecoderConv(pose_dim, n_frames)

    poses = torch.randn((4, n_frames, pose_dim))
    feat = encoder(poses)
    recon_poses = decoder(feat)

    print('input', poses.shape)
    print('feat', feat.shape)
    print('output', recon_poses.shape)
