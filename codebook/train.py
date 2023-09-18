'''
pip install easydict
pip install configargparse
'''

import pdb
# import nni

import logging
logging.getLogger().setLevel(logging.INFO)

from torch.utils.data import DataLoader
from data_loader.lmdb_data_loader import TrinityDataset
import time
import torch
import torch.nn as nn
import yaml
from pprint import pprint
from easydict import EasyDict
from configs.parse_args import parse_args
from models.vqvae import VQVAE_ulr
import os
from torch import optim
import itertools
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

# params = nni.get_next_parameter()
args = parse_args()
mydevice = torch.device('cuda:' + args.gpu)


def evaluate_testset(model, test_data_loader):
    start = time.time()
    model = model.eval()
    # tot_euclidean_error = 0
    # tot_eval_nums = 0
    upper_euclidean_errors = []
    lower_euclidean_errors = []
    root_euclidean_errors = []
    joint_channel = 16
    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            upper, lower, root, _, _ = data  # （batch, 40, 135）
            upper, lower, root = upper.to(mydevice), lower.to(mydevice), root.to(mydevice)
            # pose_seq_eval = target_vec.to(mydevice)
            b, t, c = upper.size()
            upper_, lower_, root_, _, _, _ = model(upper, lower, root)
            diff_upper = (upper_ - upper).view(b, t, c // joint_channel, joint_channel)
            diff_lower = (lower_ - lower).view(b, t, c // joint_channel // 2, joint_channel)
            diff_root = (root_ - root).view(b, t, c // joint_channel // 4, joint_channel)
            # tot_euclidean_error += torch.mean(torch.sqrt(torch.sum(diff_upper ** 2, dim=3)))
            # tot_euclidean_error += torch.mean(torch.sqrt(torch.sum(diff_lower ** 2, dim=3)))
            # tot_euclidean_error += torch.mean(torch.sqrt(torch.sum(diff_root ** 2, dim=3)))
            # tot_eval_nums += 1
            upper_euclidean_errors.append(torch.mean(torch.sqrt(torch.sum(diff_upper ** 2, dim=3))))
            lower_euclidean_errors.append(torch.mean(torch.sqrt(torch.sum(diff_lower ** 2, dim=3))))
            root_euclidean_errors.append(torch.mean(torch.sqrt(torch.sum(diff_root ** 2, dim=3))))
        # print(tot_euclidean_error / (tot_eval_nums * 1.0))
        print('generation took {:.2} s'.format(time.time() - start))
    model.train()
    return torch.mean(torch.stack(upper_euclidean_errors)).data.cpu().numpy(), \
              torch.mean(torch.stack(lower_euclidean_errors)).data.cpu().numpy(), \
                torch.mean(torch.stack(root_euclidean_errors)).data.cpu().numpy()


def main(args):
    # dataset
    train_dataset = TrinityDataset(args.train_data_path,
                                   n_poses=args.n_poses,
                                   subdivision_stride=args.subdivision_stride,
                                   pose_resampling_fps=args.motion_resampling_framerate)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True)

    val_dataset = TrinityDataset(args.val_data_path,
                                       n_poses=args.n_poses,
                                       subdivision_stride=args.subdivision_stride,
                                       pose_resampling_fps=args.motion_resampling_framerate)
    test_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                             shuffle=False, drop_last=True, num_workers=args.loader_workers, pin_memory=True)

    logging.info('len of train loader:{}, len of test loader:{}'.format(len(train_loader), len(test_loader)))

    model = VQVAE_ulr(args.VQVAE, 7 * 16)  # n_joints * n_chanels
    # model = VQVAE_ulr_easy()
    model = nn.DataParallel(model, device_ids=[eval(i) for i in config.no_cuda])
    model = model.to(mydevice)

    best_val_loss = (1e+2, 0)  # value, epoch

    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    # optimizer = optim.Adam(itertools.chain(model.module.parameters()), lr=args.lr, betas=args.betas)
    optimizer = optim.Adam(model.module.parameters(), lr=args.lr, betas=args.betas)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    updates = 0
    total = len(train_loader)

    tb_path = args.name + '_' + str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    tb_writer = SummaryWriter(log_dir=str(Path(args.model_save_path).parent / 'tensorboard_runs' / tb_path))


    for epoch in range(1, args.epochs + 1):
        logging.info(f'Epoch: {epoch}')
        i = 0
        upper_mean, lower_mean, root_mean = evaluate_testset(model, test_loader)
        logging.info('upper mean on validation: {:.3f}, lower mean on validation: {:.3f}, '
                     'root mean on validation: {:.3f}'.format(upper_mean, lower_mean, root_mean))
        # tb_writer.add_scalar('upper mean/validation', upper_mean, epoch)
        total_loss = (upper_mean + lower_mean + root_mean) / 3.0
        is_best = total_loss < best_val_loss[0]
        if is_best:
            logging.info(' *** BEST VALIDATION LOSS : {:.3f}'.format(total_loss))
            best_val_loss = (total_loss, epoch)
        else:
            logging.info(' best validation loss so far: {:.3f} at EPOCH {}'.format(best_val_loss[0], best_val_loss[1]))

        if is_best or (epoch % args.save_per_epochs == 0):
            if is_best:
                save_name = '{}/{}_checkpoint_best.bin'.format(args.model_save_path, args.name)
            else:
                save_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.model_save_path, args.name, epoch)

            torch.save({
                'args': args, "epoch": epoch, 'model_dict': model.state_dict()
            }, save_name)
            logging.info('Saved the checkpoint')

        # train model
        model = model.train()
        start = datetime.now()
        for batch_i, batch in enumerate(train_loader, 0):
            upper, lower, root, _, _ = batch
            upper, lower, root = upper.to(mydevice), lower.to(mydevice), root.to(mydevice)
            # pose_seq = target_vec.to(mydevice)      # (b, 240, 15*3)
            optimizer.zero_grad()
            _, _, _, loss1, loss2, loss3 = model(upper, lower, root)
            # write to tensorboard
            # tb_writer.add_scalar('loss' + '/train', loss, updates)
            # for key in metrics.keys():
            #     tb_writer.add_scalar(key + '/train', metrics[key], updates)
            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer.step()
            # log
            stats = {'updates': updates, 'loss1': loss1.item(), 'loss2': loss2.item(), 'loss3': loss3.item()}
            stats_str = ' '.join(f'{key}[{val:.8f}]' for key, val in stats.items())
            i += 1
            remaining = str((datetime.now() - start) / i * (total - i))
            remaining = remaining.split('.')[0]
            logging.info(f'> epoch [{epoch}] updates[{i}] {stats_str} eta[{remaining}]')
            # if i == total:
            #     logging.debug('\n')
            #     logging.debug(f'elapsed time: {str(datetime.now() - start).split(".")[0]}')
            updates += 1
        schedular.step()
    tb_writer.close()
    # print best losses
    logging.info('--------- Final best loss values ---------')
    logging.info('diff mean: {:.3f} at EPOCH {}'.format(best_val_loss[0], best_val_loss[1]))


if __name__ == '__main__':
    '''
    pip install nni
    cd codebook/
    windows: ssh -p 22 -L 8080:127.0.0.1:8080 yangsc21@server15.mjrc.ml (f8550a408967d2dc99a180893fe0b007)
    python train.py --config=./configs/codebook.yml --train --gpu 2
    '''

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    config = EasyDict(config)
    config.no_cuda = config.gpu
    main(config)
