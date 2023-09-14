import os
import pdb

import numpy as np
import yaml
from pprint import pprint
from easydict import EasyDict
import torch
from configs.parse_args import parse_args
import glob
import sys
[sys.path.append(i) for i in ['.', './conditional_gpt']]
from Hierarchical_XTransformer import Hierarchical_XTransformer, cascade_XTransformer, XTransformer_GRU, XTransformer_GPT, GPT_GRU
import torch.nn as nn
import math

args = parse_args()
mydevice = torch.device('cuda:' + args.gpu)

with open(args.config) as f:
    config = yaml.safe_load(f)

for k, v in vars(args).items():
    config[k] = v
# pprint(config)

config = EasyDict(config)


def main(args, wav_code, model, save_path, prefix=None, max_codes=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    clip_length = wav_code.shape[0]
    unit_time = int(75 * args.n_poses / args.motion_resampling_framerate)        # 4 * 75
    if clip_length < unit_time:
        num_subdivision = 1
    else:
        num_subdivision = math.ceil((clip_length - unit_time) / unit_time) + 1

    if max_codes is not None and num_subdivision >= int(max_codes / args.n_poses):
        num_subdivision = int(max_codes / args.n_poses)

    print('num_subdivision: {}, unit_time: {}, clip_length: {}'.format(num_subdivision, unit_time, clip_length))

    begin_token_1 = torch.tensor(torch.randint(0, 512, (1, 1))).to(mydevice)
    begin_token_2 = torch.tensor(torch.randint(0, 512, (1, 1))).to(mydevice)

    with torch.no_grad():
        code = []
        motion = []
        for i in range(0, num_subdivision):
            if i == 0:
                xs_1 = torch.randint(1, 512, (1, 1)).to(mydevice)
                hidden = None
            else:
                xs_1 = out_zs[:, -1:]
            start_time = i * unit_time

            # prepare pose input
            audio_start = math.floor(start_time)
            audio_end = audio_start + unit_time
            in_audio = wav_code[audio_start:audio_end]
            if len(in_audio) < unit_time:
                in_audio = np.pad(in_audio, [(0, unit_time - len(in_audio)), (0, 0)], mode='constant')

            in_audio = torch.tensor(in_audio).unsqueeze(0).to(mydevice)
            # in_audio = in_audio.reshape(in_audio.shape[0], -1)
            # out_zs = model.module.generate(seq_in=in_audio, seq_out_start=begin_token, seq_len=args.n_poses * 3)

            # out_zs = model.module.generate(in_audio, begin_token_1, begin_token_2)
            # out_zs = model.module.generate(in_audio)

            out_zs, out_lower, hidden = model.module.generate(in_audio, xs_1, hidden)
            code.append(out_zs[0].squeeze(0).data.cpu().numpy())
            motion.append(out_lower[0].squeeze(0).data.cpu().numpy())

    out_code = np.vstack(code).flatten()
    lower_motion = np.vstack(motion)
    lower_motion = np.expand_dims(lower_motion, 0).transpose((0, 2, 1))
    print(out_code.shape, lower_motion.shape)
    np.save(os.path.join(save_path, 'code_upper_' + prefix + '.npy'), out_code)
    np.save(os.path.join(save_path, 'motion_lower_' + prefix + '.npy'), lower_motion)


if __name__ == '__main__':
    '''
    python inference.py --config=./configs/pretrain.yml --train --no_cuda 3 --gpu 3
    '''
    wav_code_path = './dataset/Trinity/audio_clips/Recording_006.npy'
    model_path = './result/GPT_GRU_block_size_again/XTransformer_checkpoint_250.bin'
    save_path = './result/inference'

    # wav_code = np.random.randint(0, 1024, size=(580, 2))
    wav_code = np.load(wav_code_path)
    print('wav code shape: ', wav_code.shape)

    # model = Hierarchical_XTransformer(batch_size=1, n_poses=15, mydevice=mydevice)
    # model = cascade_XTransformer(batch_size=1, n_poses=15, mydevice=mydevice)
    # model = XTransformer_GRU(batch_size=1, n_poses=15, mydevice=mydevice)
    # model = XTransformer_GPT(batch_size=1, n_poses=15, mydevice=mydevice)

    model = GPT_GRU(batch_size=1, n_poses=15, mydevice=mydevice)
    model = nn.DataParallel(model, device_ids=[eval(i) for i in config.no_cuda])
    model = model.to(mydevice)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_dict'])
    model = model.eval()

    name = os.path.split(wav_code_path)[1][:-4]
    main(config, wav_code, model, save_path, prefix=name, max_codes=15/4*32)     # 32s
