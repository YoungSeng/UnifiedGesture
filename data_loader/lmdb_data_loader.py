import logging
import os
import pdb

import numpy as np
import lmdb as lmdb
import torch
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

import sys
[sys.path.append(i) for i in ['.', '..']]

from data_loader.data_preprocessor import DataPreprocessor
import pyarrow


def word_seq_collate_fn(data):
    """ collate function for loading word sequences in variable lengths """
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # separate source and target sequences
    word_seq, poses_seq, audio, aux_info = zip(*data)

    # merge sequences
    words_lengths = torch.LongTensor([len(x) for x in word_seq])
    word_seq = pad_sequence(word_seq, batch_first=True).long()

    poses_seq = default_collate(poses_seq)
    audio = default_collate(audio)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    return word_seq, words_lengths, poses_seq, audio, aux_info


class TrinityDataset(Dataset):
    def __init__(self, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps):
        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.lang_model = None

        logging.info("Reading data '{}'...".format(lmdb_dir))
        preloaded_dir = lmdb_dir + '_cache'
        if not os.path.exists(preloaded_dir):
            data_sampler = DataPreprocessor(lmdb_dir, preloaded_dir, n_poses,
                                            subdivision_stride, pose_resampling_fps)
            data_sampler.run()
        else:
            logging.info('Found pre-loaded samples from {}'.format(preloaded_dir))

        # init lmdb
        # map_size = 1024 * 20  # in MB
        # map_size <<= 20  # in B
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)      # default 10485760
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)

            sample = pyarrow.deserialize(sample)
            speech_seq, gesture_seq, lower_seq, aux_info = sample

        # to tensors
        speech_seq = torch.from_numpy(speech_seq).long()
        gesture_seq = torch.from_numpy(gesture_seq).long()
        lower_seq = torch.from_numpy(lower_seq).float()

        return speech_seq, gesture_seq, lower_seq, aux_info


if __name__ == '__main__':
    '''
    python data_loader/lmdb_data_loader.py --config=./configs/pretrain.yml --train --no_cuda 3 --gpu 3
    '''

    from configs.parse_args import parse_args
    import os
    import yaml
    from pprint import pprint
    from easydict import EasyDict
    from torch.utils.data import DataLoader

    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    args = EasyDict(config)

    train_dataset = TrinityDataset(args.train_data_path,
                                   n_poses=args.n_poses,
                                   subdivision_stride=args.subdivision_stride,
                                   pose_resampling_fps=args.motion_resampling_framerate)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True)

    print(len(train_loader))        # 247
    for batch_i, batch in enumerate(train_loader, 0):
        speech, gesture, aux = batch        # [128, 300, 2] 0-1023, [128, 15, 3] 0-511
        print(batch_i)
        pdb.set_trace()
