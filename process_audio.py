import os
import pdb
import torchaudio
from conditional_gpt.encodec.utils import convert_audio
import numpy as np
import torch


def clip_audio(src_audio_path, src_motion_path, save_path, mode):

    from conditional_gpt.encodec import EncodecModel
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(1.5)

    if not os.path.exists(save_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(save_path)  # makedirs 创建文件时如果路径不存在会创建这个路径
    if mode == 'ZEGGS':
        flag = 461
        audio_sample_rate = 16000
    elif mode == 'Trinity':
        flag = 343
        audio_sample_rate = 48000
    for bvh_file in os.listdir(src_motion_path):
        print(bvh_file)
        with open(os.path.join(src_motion_path, bvh_file), 'r') as f:
            content = f.readlines()
        motion_frames = eval(content[flag].strip().split(':')[-1])
        motion_frames = motion_frames // 4 * 4      # retargeting 4x
        length = motion_frames * audio_sample_rate // 30

        audio_file, sr = torchaudio.load(os.path.join(src_audio_path, bvh_file[:-4] + '.wav'))

        print(sr, motion_frames, length, audio_file.shape[1])
        audio_file = audio_file[:, :length]
        # assert audio_file.shape[1] == length

        wav = convert_audio(audio_file, sr, model.sample_rate, model.channels)
        wav = wav.unsqueeze(0)
        with torch.no_grad():
            encoded_frames = model.encode(wav)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]        # [1, 8, 1500]
        codes = codes.squeeze(0).transpose(0, 1).numpy()
        np.save(os.path.join(save_path, bvh_file[:-4] + '.npy'), codes)


def process_audio(src_audio_path, src_motion_path, save_path, mode):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if 'ZEGGS' in mode:
        flag = 461
        audio_sample_rate = 16000
    elif 'Trinity' in mode:
        flag = 343
        audio_sample_rate = 48000
    if 'valid' not in mode:
        for bvh_file in os.listdir(src_motion_path):
            print(bvh_file)
            with open(os.path.join(src_motion_path, bvh_file), 'r') as f:
                content = f.readlines()
            motion_frames = eval(content[flag].strip().split(':')[-1])
            motion_frames = motion_frames // 4 * 4      # retargeting 4x
            length = motion_frames * audio_sample_rate // 30

            audio_file, sr = torchaudio.load(os.path.join(src_audio_path, bvh_file[:-4] + '.wav'))

            print(sr, motion_frames, length, audio_file.shape[1])
            audio_file = audio_file[:, :length]
            # assert audio_file.shape[1] == length

            wav = convert_audio(audio_file, sr, 16000, 1)

            np.save(os.path.join(save_path, bvh_file[:-4] + '.npy'), wav.numpy())
    else:
        for wav_file in os.listdir(src_audio_path):
            print(wav_file)
            audio_file, sr = torchaudio.load(os.path.join(src_audio_path, wav_file))
            wav = convert_audio(audio_file, sr, 16000, 1)
            np.save(os.path.join(save_path, wav_file[:-4] + '.npy'), wav.numpy())


if __name__ == '__main__':
    '''
    python process_audio.py
    '''
    trinity_motion_path = './retargeting/datasets/Trinity_ZEGGS/Trinity/'
    zeggs_motion_path = './retargeting/datasets/Trinity_ZEGGS/ZEGGS/'
    trinity_audio_path = './dataset/Trinity/audio/'
    zeggs_audio_path = './dataset/ZEGGS/clean/'
    # trinity_audio_clip_path = './dataset/Trinity/audio_clips/'
    # zeggs_audio_clip_path = './dataset/ZEGGS/audio_clips/'
    trinity_audio_proceed_path = './dataset/Trinity/all_speech/'
    zeggs_audio_proceed_path = './dataset/ZEGGS/all_speech/'

    # 1. clip audio
    # clip_audio(zeggs_audio_path, zeggs_motion_path, zeggs_audio_clip_path, 'ZEGGS')
    # clip_audio(trinity_audio_path, trinity_motion_path, trinity_audio_clip_path, 'Trinity')

    # 2. process training audio
    process_audio(zeggs_audio_path, zeggs_motion_path, zeggs_audio_proceed_path, 'ZEGGS')
    process_audio(trinity_audio_path, trinity_motion_path, trinity_audio_proceed_path, 'Trinity')

    # 3. process valid audio
    # trinity_audio_proceed_path = './dataset/Trinity/valid_speech/'
    # zeggs_audio_proceed_path = './dataset/ZEGGS/valid_speech/'
    # process_audio('../ZEGGS/valid/', zeggs_motion_path, zeggs_audio_proceed_path, 'ZEGGS_valid')
    # process_audio('../trinity/test/', trinity_motion_path, trinity_audio_proceed_path, 'Trinity_valid')
