""" create data samples """
import pdb

import lmdb
import math
import numpy as np
import pyarrow


import torch
import torch.nn.functional as F

def wavlm_init(device='cuda:2'):
    import sys
    [sys.path.append(i) for i in ['./WavLM']]
    from WavLM import WavLM, WavLMConfig
    wavlm_model_path = './wavlm_cache/WavLM-Large.pt'
    # load the pre-trained checkpoints
    checkpoint = torch.load(wavlm_model_path, map_location=torch.device('cpu'))
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    device = torch.device(device)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def wav2wavlm(model, wav_input_16khz, device='cuda:2'):
    with torch.no_grad():
        device = torch.device(device)
        # print(wav_input_16khz.shape)
        wav_input_16khz = torch.from_numpy(wav_input_16khz).to(device)
        # wav_input_16khz = wav_input_16khz.to(device).unsqueeze(0)
        rep = model.extract_features(wav_input_16khz)[0]
        del wav_input_16khz
        rep = F.interpolate(rep.transpose(1, 2), size=36, align_corners=True, mode='linear').transpose(1, 2)
        return rep.squeeze().cpu().detach().data.cpu().numpy()


class DataPreprocessor:
    def __init__(self, clip_lmdb_dir, out_lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, n_codes=30):
        self.n_codes = n_codes
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps

        self.src_lmdb_env = lmdb.open(clip_lmdb_dir, readonly=True, lock=False)
        with self.src_lmdb_env.begin() as txn:
            self.n_videos = txn.stat()['entries']

        self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * 16000)

        # create db for samples
        map_size = 1024 * 1024 * 9  # in TB
        map_size <<= 20  # in B
        self.dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=map_size)
        self.n_out_samples = 0

        self.model = wavlm_init()

    def run(self):
        src_txn = self.src_lmdb_env.begin(write=False)

        # sampling and normalization
        cursor = src_txn.cursor()
        for key, value in cursor:
            video = pyarrow.deserialize(value)
            vid = video['vid']
            clips = video['clips']
            for clip_idx, clip in enumerate(clips):
                self._sample_from_clip(vid, clip)

        # print stats
        with self.dst_lmdb_env.begin() as txn:
            print('no. of samples: ', txn.stat()['entries'])

        # close db
        self.src_lmdb_env.close()
        self.dst_lmdb_env.sync()
        self.dst_lmdb_env.close()

    def _sample_from_clip(self, vid, clip):
        clip_skeleton = clip['full_body']
        clip_audio_raw = clip['audio_wav']
        clip_styles_raw = clip['style']
        clip_upper_code_raw = clip['upper_code']
        # clip_skeleton_vel = clip_skeleton[:-1] - clip_skeleton[1:]
        # clip_skeleton_acc = clip_skeleton_vel[:-1] - clip_skeleton_vel[1:]
        # clip_skeleton_vel = np.pad(clip_skeleton_vel, ((1, 0), (0, 0)), 'constant')
        # clip_skeleton_acc = np.pad(clip_skeleton_acc, ((2, 0), (0, 0)), 'constant')

        # divide
        aux_info = []
        sample_skeletons_list = []
        # sample_skeletons_list_vel = []
        # sample_skeletons_list_acc = []
        sample_style_list = []
        sample_codes_list = []
        sample_wavlm_list = []

        MINLEN = len(clip_skeleton)

        num_subdivision = math.floor((MINLEN - self.n_poses) / self.subdivision_stride)  # floor((K - (N+M)) / S) + 1

        for i in range(num_subdivision):
            start_idx = i * self.subdivision_stride
            fin_idx = start_idx + self.n_poses

            sample_skeletons = clip_skeleton[start_idx:fin_idx]
            # sample_skeletons_vel = clip_skeleton_vel[start_idx:fin_idx]
            # sample_skeletons_acc = clip_skeleton_acc[start_idx:fin_idx]
            sample_code = clip_upper_code_raw[start_idx//2:fin_idx//2]

            subdivision_start_time = start_idx / self.skeleton_resampling_fps
            subdivision_end_time = fin_idx / self.skeleton_resampling_fps

            # raw audio
            audio_start = math.floor(start_idx / len(clip_skeleton) * len(clip_audio_raw))
            audio_end = audio_start + self.audio_sample_length
            sample_audio = clip_audio_raw[:, audio_start:audio_end]
            sample_wavlm = wav2wavlm(self.model, sample_audio)      # (1, 68266)

            motion_info = {'vid': vid,
                           'start_frame_no': start_idx,
                           'end_frame_no': fin_idx,
                           'start_time': subdivision_start_time,
                           'end_time': subdivision_end_time}

            sample_skeletons_list.append(sample_skeletons)
            # sample_skeletons_list_vel.append(sample_skeletons_vel)
            # sample_skeletons_list_acc.append(sample_skeletons_acc)
            sample_codes_list.append(sample_code)
            sample_wavlm_list.append(sample_wavlm)
            sample_style_list.append(clip_styles_raw)

            aux_info.append(motion_info)

        if len(sample_skeletons_list) > 0:
            with self.dst_lmdb_env.begin(write=True) as txn:
                # for poses, vel, acc, code, style, wavlm, aux in zip(sample_skeletons_list, sample_skeletons_list_vel, sample_skeletons_list_acc, sample_codes_list, sample_style_list, sample_wavlm_list, aux_info):
                for poses, code, style, wavlm, aux in zip(sample_skeletons_list, sample_codes_list, sample_style_list, sample_wavlm_list, aux_info):

                    poses = np.asarray(poses)
                    # vel = np.asarray(vel)
                    # acc = np.asarray(acc)
                    code = np.asarray(code)
                    style = np.asarray(style)
                    wavlm = np.asarray(wavlm)

                    # save
                    k = '{:010}'.format(self.n_out_samples).encode('ascii')
                    # v = [poses, vel, acc, code, style, wavlm, aux]
                    v = [poses, code, style, wavlm]
                    v = pyarrow.serialize(v).to_buffer()
                    txn.put(k, v)
                    self.n_out_samples += 1

