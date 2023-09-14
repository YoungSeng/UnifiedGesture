import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from embedding_net import EmbeddingNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def train_iter(target_data, net, optim):
    # zero gradients
    optim.zero_grad()

    # reconstruction loss
    feat, recon_data = net(target_data)
    recon_loss = F.l1_loss(recon_data, target_data, reduction='none')
    recon_loss = torch.mean(recon_loss, dim=(1, 2))

    if True:  # use pose diff
        target_diff = target_data[:, 1:] - target_data[:, :-1]
        recon_diff = recon_data[:, 1:] - recon_data[:, :-1]
        recon_loss += torch.mean(F.l1_loss(recon_diff, target_diff, reduction='none'), dim=(1, 2))

    recon_loss = torch.sum(recon_loss)

    recon_loss.backward()
    optim.step()

    ret_dict = {'loss': recon_loss.item()}
    return ret_dict


def make_tensor(path, n_frames, stride=None):

    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*.npy'))
    else:
        files = [path]

    samples = []
    stride = n_frames // 2 if stride is None else stride
    for file in files:
        data = np.load(file).reshape(-1, 33 * 3)
        for i in range(0, len(data) - n_frames, stride):
            sample = data[i:i+n_frames]
            samples.append(sample)

    return torch.Tensor(samples)


def main(n_frames):
    # dataset
    train_dataset = TensorDataset(make_tensor('/ceph/hdd/yangsc21/Python/My_3/GT_Gesture_npy/', n_frames, stride=10))
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=False)

    # train
    loss_meters = [AverageMeter('loss')]

    # interval params
    print_interval = int(len(train_loader) / 5)

    # init model and optimizer
    pose_dim = 33 * 3
    generator = EmbeddingNet(pose_dim, n_frames).to(device)
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.0004, betas=(0.5, 0.999))

    # training
    for epoch in range(100):
        for iter_idx, target in enumerate(train_loader, 0):
            target = target[0]
            batch_size = target.size(0)
            target_vec = target.to(device)
            loss = train_iter(target_vec, generator, gen_optimizer)

            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | '.format(epoch, iter_idx + 1)
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                print(print_summary)

    # save model
    gen_state_dict = generator.state_dict()
    save_name = f'output/model_checkpoint_{n_frames}.bin'
    torch.save({'pose_dim': pose_dim, 'n_frames': n_frames, 'gen_dict': gen_state_dict}, save_name)


if __name__ == '__main__':
    '''
    cd ./genea_numerical_evaluations/FGD
    python train_AE.py
    '''
    n_frames = 120  # 30, 60, 90
    main(n_frames)
