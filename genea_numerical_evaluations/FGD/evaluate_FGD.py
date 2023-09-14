import os

import numpy as np
import torch

from embedding_space_evaluator import EmbeddingSpaceEvaluator
from train_AE import make_tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_fgd(fgd_evaluator, gt_data, test_data):
    fgd_evaluator.reset()

    fgd_evaluator.push_real_samples(gt_data)
    fgd_evaluator.push_generated_samples(test_data)
    fgd_on_feat = fgd_evaluator.get_fgd(use_feat_space=True)
    # fdg_on_raw = fgd_evaluator.get_fgd(use_feat_space=False)
    fdg_on_raw = 0
    return fgd_on_feat, fdg_on_raw


def exp_base(chunk_len):
    # AE model
    ae_path = f'output/model_checkpoint_{chunk_len}.bin'
    fgd_evaluator = EmbeddingSpaceEvaluator(ae_path, chunk_len, device)

    # load GT data
    gt_data = make_tensor("/ceph/hdd/yangsc21/Python/My_3/GT_Gesture_npy/", chunk_len).to(device)

    # load generated data
    generated_data_path = 'data'
    # folders = sorted([f.path for f in os.scandir(generated_data_path) if f.is_dir() and 'Cond_' in f.path])

    folders = ["/ceph/hdd/yangsc21/Python/My_3/wTrinity_npy/"]

    print(f'----- Experiment (motion chunk length: {chunk_len}) -----')
    print('FGDs on feature space and raw data space')
    for folder in folders:
        test_data = make_tensor(folder, chunk_len).to(device)
        fgd_on_feat, fgd_on_raw = run_fgd(fgd_evaluator, gt_data, test_data)
        print(f'{os.path.basename(folder)}: {fgd_on_feat:8.3f}, {fgd_on_raw:8.3f}')
    print()


# def exp_per_seq(chunk_len, stride=1):
#     n_test_seq = 10
#
#     # AE model
#     ae_path = f'output/model_checkpoint_{chunk_len}.bin'
#     fgd_evaluator = EmbeddingSpaceEvaluator(ae_path, chunk_len, device)
#
#     # run
#     print(f'----- Experiment (motion chunk length: {chunk_len}, stride: {stride}) -----')
#     print('FGDs on feature space and raw data space for each system and each test speech sequence')
#
#     results = []
#     for i in range(n_test_seq):
#         name = f'TestSeq{i+1:03d}'
#
#         # load GT data
#         gt_data = make_tensor(os.path.join('data/GroundTruth', name + '.npz'), chunk_len, stride=stride)
#         gt_data = gt_data.to(device)
#         # print(gt_data.shape)
#
#         # load generated data
#         test_data = make_tensor(os.path.join(f'data/Cond_{system_name}', name + '.npz'), chunk_len, stride=stride)
#         test_data = test_data.to(device)
#         # print(test_data.shape)
#
#         # calculate fgd
#         fgd_on_feat, fgd_on_raw = run_fgd(fgd_evaluator, gt_data, test_data)
#         print(f' {name}: {fgd_on_feat:8.3f}, {fgd_on_raw:8.3f}')
#         results.append([fgd_on_feat, fgd_on_raw])
#     results = np.array(results)
#     print(f' M=({np.mean(results[:, 0])}, {np.mean(results[:, 1])}), SD=({np.std(results[:, 0])}, {np.std(results[:, 1])})')


if __name__ == '__main__':
    '''
    python evaluate_FGD.py
    '''
    # calculate fgd per system
    exp_base(120)

    # calculate fgd per system per test speech sequence
    # exp_per_seq(120, stride=1)
