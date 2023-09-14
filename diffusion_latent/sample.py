import pdb
import sys
from utils_.model_util import create_gaussian_diffusion, load_model_wo_clip
[sys.path.append(i) for i in ['.', '..', '../model', '../codebook']]
from model.mdm import MDM
import os
from datetime import datetime
import librosa
import numpy as np
import yaml
from pprint import pprint
from configs.parse_args import parse_args
import torch
import torch.nn.functional as F
from easydict import EasyDict
import math
import glob


args = parse_args()
mydevice = torch.device('cuda:' + args.gpu)
torch.cuda.set_device(int(args.gpu))

batch_size = 1


style2onehot = {
'Happy':[1, 0, 0, 0, 0, 0, 0],
'Sad':[0, 1, 0, 0, 0, 0, 0],
'Neutral':[0, 0, 1, 0, 0, 0, 0],
'Old':[0, 0, 0, 1, 0, 0, 0],
'Angry':[0, 0, 0, 0, 1, 0, 0],
'Relaxed':[0, 0, 0, 0, 0, 1, 0],
'Still':[0, 0, 0, 0, 0, 0, 1]
}

feature_dim = 16 * 7 * 3 + 27        #  + 27
seed = 6        # 4
n_poses = 36
dim = 256           # 256
MAX_LEN = 0       # 0 for generate all frames, 1 * 32 used debug


def vqvae_init(device=torch.device('cuda:2')):
    import torch.nn as nn
    from codebook.models.vqvae import VQVAE_ulr
    model = VQVAE_ulr(config.VQVAE, 7 * 16)  # n_joints * n_chanels
    model_path = "../codebook/result/train_codebook_upper_lower_root_downsample2/codebook_checkpoint_best.bin"
    # model_path = "/ceph/home/wangzl21/Projects/My_3/deep-motion-editing/diffusion_latent/experiments/reinforce_diffusion_seed0/ckpt/vqvae_epoch_1.bin"
    model = nn.DataParallel(model, device_ids=[eval(i) for i in config.no_cuda])
    model = model.to(device)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_dict'])
    # model.load_state_dict(checkpoint['model'])       # zerlin 20230428
    model = model.eval()
    return model


def latent2code(model, source_upper, save_path="../codebook/result/inference/Trinity", prefix = 'train_codebook_bvh2upper_lower_root_again', w_grad=False, code=None):
    from VisualizeCodebook import main_2
    # print(source_upper.shape)       # (len, 64)
    recon, code, out_distance = main_2(source_upper, model, save_path, prefix, w_grad, code=code)
    return recon, code, out_distance


def wavlm_init(device=torch.device('cuda:2')):
    import sys
    [sys.path.append(i) for i in ['./WavLM']]
    from WavLM import WavLM, WavLMConfig
    wavlm_model_path = './wavlm_cache/WavLM-Large.pt'
    # load the pre-trained checkpoints
    checkpoint = torch.load(wavlm_model_path, map_location=torch.device('cpu'))
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    # model.eval()
    return model


def wav2wavlm(model, wav_input_16khz, device=torch.device('cuda:2')):
    with torch.no_grad():
        wav_input_16khz = wav_input_16khz.to(device)
        rep = model.extract_features(wav_input_16khz)[0]
        rep = F.interpolate(rep.transpose(1, 2), size=n_poses, align_corners=True, mode='linear').transpose(1, 2)
    return rep


def create_model_and_diffusion():
    model = MDM(modeltype='', njoints=feature_dim, nfeats=1, translation=True, pose_rep='rot6d', glob=True,
                glob_rot=True, cond_mode='cross_local_attention3_style1', action_emb='tensor', audio_feat='wavlm',
                arch='trans_enc', latent_dim=dim, n_seed=seed, cond_mask_prob=0.1)
    diffusion = create_gaussian_diffusion()
    return model, diffusion


def inference(args, save_dir, save_name, wavlm_model, audio, sample_fn, model, n_frames=0, smoothing=False, SG_filter=False, minibatch=False, skip_timesteps=False, n_seed=seed, style=None, seed=123456, time_step=False, time=None, return_seed=False, w_grad=False, save_result=False):

    torch.manual_seed(seed)

    if n_frames == 0:
        n_frames = int(audio.shape[0] * 7.5 // 16000)
    if minibatch:
        stride_poses = args.n_poses - n_seed        # 32
        if n_frames < stride_poses:
            num_subdivision = 1
        else:
            num_subdivision = math.floor(n_frames / stride_poses)
            n_frames = num_subdivision * stride_poses       # 8
            print(
                '{}, {}, {}'.format(num_subdivision, stride_poses, n_frames))

    audio = audio[:num_subdivision * math.floor(stride_poses * 16000 / 7.5)]             # (1, 76800)

    model_kwargs_ = {'y': {}}
    model_kwargs_['y']['mask'] = (torch.zeros([1, 1, 1, n_frames]) < 1).to(mydevice)
    model_kwargs_['y']['style'] = torch.as_tensor([style]).float().to(mydevice)
    model_kwargs_['y']['mask_local'] = torch.ones(1, args.n_poses).bool().to(mydevice)

    if minibatch:
        audio_reshape = torch.from_numpy(audio).to(torch.float32).reshape(num_subdivision, math.floor(stride_poses * 16000 / 7.5)).to(mydevice).transpose(0, 1)     #        # mfcc[:, :-2]
        shape_ = (1, model.njoints, model.nfeats, args.n_poses)
        out_list = []
        wavlm_result = []
        time_step_list = []
        for i in range(0, num_subdivision):
            print(i, num_subdivision)
            model_kwargs_['y']['audio'] = audio_reshape[:, i:i + 1]
            if time_step:
                if type(time) is np.ndarray:
                    t = time[i]
                else:
                    t = np.random.randint(1, 1000 + 1)
                skip_t = t - 1
                time_step_list.append(t)
            else:
                t = 1000
                skip_t = 0
            if i == 0:
                if n_seed != 0:
                    pad_zeros = torch.zeros([int(n_seed * 16000 / 7.5), 1]).to(mydevice)        # wavlm dims are 1024
                    model_kwargs_['y']['audio'] = torch.cat((pad_zeros, model_kwargs_['y']['audio']), 0)
                    model_kwargs_['y']['seed'] = torch.zeros([1, feature_dim, 1, n_seed]).to(mydevice)
            else:
                if n_seed != 0:
                    pad_audio = audio_reshape[-int(n_seed * 16000 / 7.5):, i - 1:i]
                    model_kwargs_['y']['audio'] = torch.cat((pad_audio, model_kwargs_['y']['audio']), 0)
                    # out_list[-1].shape, [1, 224, 1, 32]
                    # if feature_dim == 1:
                    #     model_kwargs_['y']['seed'] = out_list[-1][..., -n_seed:].to(mydevice)
                    # elif feature_dim >= 2:
                    #     model_kwargs_['y']['seed'] = torch.cat((out_list[-1][:, :16 * 7][..., -n_seed:].to(mydevice), torch.zeros([1, 16 * 7 * (feature_dim - 1), 1, n_seed]).to(mydevice)), dim=1)
                    model_kwargs_['y']['seed'] = out_list[-1][..., -n_seed:].to(mydevice)

            wavlm_feature = wav2wavlm(wavlm_model, model_kwargs_['y']['audio'].transpose(0, 1), device=mydevice)
            if w_grad:
                wavlm_feature.requires_grad = False
            model_kwargs_['y']['audio'] = wavlm_feature

            wavlm_feature_downsample = F.interpolate(wavlm_feature.transpose(1, 2), size=n_poses//2, align_corners=True, mode='linear').transpose(1, 2)
            wavlm_result.append(wavlm_feature_downsample.transpose(0, 1).detach().data.cpu().numpy())
            sample = sample_fn(
                model,
                shape_,
                clip_denoised=False,
                model_kwargs=model_kwargs_,
                skip_timesteps=skip_t,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,  # None, torch.randn(*shape_, device=mydevice)
                const_noise=False,
                cond_fn_with_grad=w_grad,
                num_timesteps=t,
            )
            # smoothing motion transition
            if len(out_list) > 0 and n_seed != 0:
                last_poses = out_list[-1][..., -n_seed:]        # [1, 336, 1, 6]
                if not return_seed:
                    out_list[-1] = out_list[-1][..., :-n_seed]  # delete last 4 frames
                    wavlm_result[-1] = wavlm_result[-1][:-n_seed//2]
                if smoothing:
                    # Extract predictions
                    last_poses_root_pos = last_poses[:, :16 * 7]        # (1, 3, 1, 8)

                    next_poses_root_pos = sample[:, :16 * 7]        # (1, 3, 1, 88)

                    root_pos = last_poses_root_pos[..., 1]      # (1, 3, 1)
                    predict_pos = next_poses_root_pos[..., 0]
                    delta_pos = (predict_pos - root_pos).unsqueeze(-1)      # # (1, 3, 1, 1)
                    sample[:, :16 * 7] = sample[:, :16 * 7] - delta_pos

                for j in range(len(last_poses)):
                    n = len(last_poses)
                    prev = last_poses[..., j]
                    next = sample[..., j]
                    sample[..., j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)
            out_list.append(sample)     # [1, 112, 1, 32]


        if not return_seed:
            out_list[-1] = out_list[-1][..., :-n_seed]
        if not w_grad:
            out_list = [i.detach().data.cpu().numpy() for i in out_list]
            out_dir_vec = np.vstack(out_list)
        else:
            out_dir_vec = torch.cat(out_list, dim=0)
        if not return_seed:
            if not w_grad:
                sampled_seq = out_dir_vec.squeeze(2).transpose(0, 2, 1).reshape(batch_size, n_frames, model.njoints)
            else:
                sampled_seq = out_dir_vec.squeeze(2).permute(0, 2, 1).reshape(batch_size, n_frames, model.njoints)
        else:
            if not w_grad:
                sampled_seq = out_dir_vec.squeeze(2).transpose(0, 2, 1).reshape(1, num_subdivision * args.n_poses, model.njoints)     # (batch, 36, 336)
            else:
                sampled_seq = out_dir_vec.squeeze(2).permute(0, 2, 1).reshape(1, num_subdivision * args.n_poses, model.njoints)

    if not w_grad:
        out_poses = sampled_seq[..., :16 * 7].transpose(0, 2, 1)  # (1, 238, 112) -> (1, 112, 238)
    else:
        out_poses = sampled_seq[..., :16 * 7].permute(0, 2, 1)
    print(out_poses.shape)
    # prefix = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    prefix = save_name
    if smoothing: prefix += '_smoothing'
    if SG_filter: prefix += '_SG'
    if minibatch: prefix += '_minibatch'
    prefix += '_%s' % (n_frames)
    prefix += '_' + str(style)
    prefix += '_' + str(seed)

    output_path = save_dir + '/' + prefix
    if save_result:
        np.save(output_path + '.npy', out_poses)

    if time_step:
        return out_poses, output_path, np.vstack(wavlm_result), np.vstack(time_step_list)        # (1, 112, len)
    else:
        return out_poses, output_path, np.vstack(wavlm_result), None        # (1, 112, len)

def main(save_dir, model_path):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # sample
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion()
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(mydevice)
    # model.eval()

    sample_fn = diffusion.p_sample_loop     # predict x_start

    # style = mfcc_path.split('/')[-1].split('_')[1]
    wavlm_model = wavlm_init(device=mydevice)
    vqvae_model = vqvae_init(mydevice)

    return sample_fn, wavlm_model, vqvae_model, model


def inference_2(args, sample_fn, wavlm_model, vqvae_model, model, audionpy_path=None, audio_path=None, audio_wav=None, time_step=False, time=None, return_seed=False, w_grad=False, save_result=False):
    if type(audio_wav) is np.ndarray and audio_path != None:
        save_name = os.path.split(audio_path)[1][:-4]
        audio_feat = audio_wav
    elif audio_path != None:
        audio_feat, fs = librosa.load(audio_path, sr=16000)
        save_name = os.path.split(audio_path)[1][:-4]
    elif audionpy_path != None:
        audio_feat = np.load(audionpy_path)[0]
        save_name = os.path.split(audionpy_path)[1][:-4]

    if '_' in save_name:
        style = save_name.split('_')[1]
        print(save_name, style)
        if style in style2onehot:
            style = np.array(style2onehot[style])     # 20230430        #  * 1.0
    else:
        style = style2onehot['Neutral']

    x, output_path, wavlm_feat, t_list = inference(args, save_dir, save_name, wavlm_model, audio_feat, sample_fn, model,
                                                   n_frames=MAX_LEN, smoothing=False, SG_filter=False, minibatch=True,
                                                   skip_timesteps=time_step, style=[0, 0, 0, 0, 0, 3, 0], seed=123456, time_step=time_step,
                                                   time=time,
                                                   return_seed=return_seed, w_grad=w_grad, save_result=save_result)      # style2onehot['Happy']

    # x = np.load("./result/Trinity/Recording_006_minibatch_96_[0, 0, 0, 1, 0, 0, 0]_123456.npy")  # (1, 112, len)
    x_ = x[0].transpose(1, 0)[..., 16 * 2:-16 * 1]  # (len, 112)
    # my_code = np.array([[101], [140], [140], [140], [140], [140], [368], [239], [239], [239], [ 16], [ 16], [ 16], [ 16], [ 16]])
    # my_code = np.array([[480, 480, 140, 140, 140, 140, 368, 239, 239, 239, 16, 16, 16, 16, 16]])

    # my_code = np.array([[480, 480, 480, 182, 272, 272, 272, 272, 272, 272, 272, 272, 16, 16, 16]])

    recon, code, distance = latent2code(vqvae_model, x_, None, None, w_grad, code=None)
    if w_grad:
        x = x.detach().data.cpu().numpy()
    y = np.concatenate((x[:, :16 * 2], recon, x[:, -16 * 1:]), axis=1)  # (1, 112, len)
    np.save(output_path + "_recon.npy", y)
    print(output_path, code)
    np.save(output_path + "_code.npy", code)
    # print(code.shape, x.shape, y.shape)
    return y, x, code, wavlm_feat, distance, t_list


def single_inference(audio, config, sample_fn, wavlm_model, vqvae_model, diffusion_model, audio_wav=None, time_step=False, time=None, return_seed=False, w_grad=False, save_result=False):
    if not w_grad:
        vqvae_model = vqvae_model.eval()
        diffusion_model = diffusion_model.eval()
    else:
        vqvae_model = vqvae_model.train()
        vqvae_model.module.freeze_drop()
        diffusion_model = diffusion_model.train()
        diffusion_model.freeze_drop()

    if type(audio_wav) is np.ndarray:
        motion_vq, motion_diff, code, wavlm_feat, distance, t_list = inference_2(config, sample_fn, wavlm_model, vqvae_model, diffusion_model,
                audio_path=audio, audio_wav=audio_wav, time_step=time_step, time=time, return_seed=return_seed, w_grad=w_grad, save_result=save_result)
    else:
        if audio.endswith('.wav'):
            motion_vq, motion_diff, code, wavlm_feat, distance, t_list = inference_2(config, sample_fn, wavlm_model, vqvae_model, diffusion_model,
                                       audio_path=audio, time_step=time_step, time=time, return_seed=return_seed, w_grad=w_grad, save_result=save_result)
        elif audio.endswith('.npy'):
            motion_vq, motion_diff, code, wavlm_feat, distance, t_list = inference_2(config, sample_fn, wavlm_model, vqvae_model, diffusion_model,
                                       audionpy_path=audio, time_step=time_step, time=time, return_seed=return_seed, w_grad=w_grad, save_result=save_result)

    return motion_vq, motion_diff, code, wavlm_feat, distance, t_list


def all_inference(ZEGGS_audio_fold, Trinity_audio_fold, ZEGGS_audio_fold_valid, Trinity_audio_fold_valid, config, sample_fn, wavlm_model, vqvae_model, diffusion_model, return_GT=False):
    train_audio_list = sorted(glob.glob(ZEGGS_audio_fold + '/*.npy') + glob.glob(Trinity_audio_fold + '/*.npy'))
    valid_audio_list = sorted(glob.glob(ZEGGS_audio_fold_valid + '/*.npy') + glob.glob(Trinity_audio_fold_valid + '/*.npy'))

    GT_list = []

    # motion_list = []
    code_list = []

    for audio in train_audio_list:
        name = os.path.split(audio)[1][:-4]
        audio_wav = np.load(audio)[0]
        print(name)
        num_subdivision = math.floor(int(audio_wav.shape[0] * 7.5 // 16000 / (config.n_poses - seed)))
        time = np.array([500] * num_subdivision)        # [1, 1000]

        motion_vq, motion_diff, code, wavlm_feat, distance, t_list = single_inference(audio, config, sample_fn, wavlm_model, vqvae_model, diffusion_model, audio_wav=audio_wav, time_step=True, time=time, return_seed=True, w_grad=False)

        # motion_list.append(motion)
        code = code.squeeze(1)
        wavlm_feat = wavlm_feat.squeeze(1)[:code.shape[0]]
        code_list.append(code)
        if return_GT:
            if name.split('_')[0] == 'Recording':
                GT_code = np.load("../dataset/Trinity/all_gesture_aux/" + name + '_upper.npy')
            else:
                GT_code = np.load("../dataset/ZEGGS/all_gesture_aux/" + name + '_upper.npy')
            GT_code = GT_code[:code.shape[0]]
            GT_list.append(GT_code)
        # break

        '''
        # np.argmin(distance, axis=1) == code
        # TODO: save wavlm_feat, code(distance) and t
        '''
        code = code.reshape(-1, config.n_poses//2)
        distance = distance.reshape(-1, config.n_poses//2, 512)
        wavlm_feat = wavlm_feat.reshape(-1, config.n_poses//2, 1024)
        pdb.set_trace()

    return code_list, GT_list


def generate_result(ZEGGS_audio_fold_valid, Trinity_audio_fold_valid, config, sample_fn, wavlm_model, vqvae_model, diffusion_model, return_GT=False):
    valid_audio_list = sorted(glob.glob(ZEGGS_audio_fold_valid + '/*.npy') + glob.glob(Trinity_audio_fold_valid + '/*.npy'))

    GT_list = []

    # motion_list = []
    code_list = []

    for audio in valid_audio_list:
        name = os.path.split(audio)[1][:-4]
        audio_wav = np.load(audio)[0]
        print(name)

        motion_vq, motion_diff, code, wavlm_feat, distance2, t_list = single_inference(audio, config, sample_fn, wavlm_model, vqvae_model, diffusion_model, audio_wav=audio_wav, time_step=False, save_result=False)

        # motion_list.append(motion)
        code = code.squeeze(1)
        wavlm_feat = wavlm_feat.squeeze(1)[:code.shape[0]]
        code_list.append(code)
        # break

        '''
        # np.argmin(distance, axis=1) == code
        # TODO: save wavlm_feat, code(distance) and t
        '''
        # code = code.reshape(-1, config.n_poses//2)
        # distance = distance.reshape(-1, config.n_poses//2, 512)
        # wavlm_feat = wavlm_feat.reshape(-1, config.n_poses//2, 1024)

        # [sys.path.append(i) for i in ['../retargeting']]
        # from retargeting.demo import latent2bvh
        np.save(save_dir + '/' + name + '_final' + '.npy', motion_vq)
        np.save(save_dir + '/' + name + '_wovq' + '.npy', motion_diff)


if __name__ == '__main__':
    '''
    /ceph/hdd/yangsc21/Python/My_3/deep-motion-editing/diffusion_latent/
    # pip install ffmpeg-normalize, see https://github.com/slhck/ffmpeg-normalize
    '''

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    # pprint(config)

    config = EasyDict(config)

    # model_path = './result/inference/new_DiffuseStyleGesture_model3_256/model001100000.pt'
    # model_path = './result/inference/new_DiffuseStyleGesture_model3_128/model001150000.pt'
    # model_path = "/ceph/home/wangzl21/Projects/My_3/deep-motion-editing/diffusion_latent/experiments/reinforce_diffusion_onlydiff_gradnorm0.1_lr1e-6_max0_seed0/ckpt/diffusion_epoch_1.pt"
    # model_path = '/ceph/home/wangzl21/Projects/My_3/deep-motion-editing/diffusion_latent/experiments/256_seed_4_aux_model001700000_reinforce_diffusion_onlydiff_gradnorm0.1_lr1e-6_max0_seed0/ckpt/diffusion_epoch_1.pt'

    # model_path = "/ceph/home/wangzl21/Projects/My_3/deep-motion-editing/diffusion_latent/experiments/256_seed_6_aux_model001700000_reinforce_diffusion_onlydiff_gradnorm0.1_lr1e-6_max0_seed0/ckpt/diffusion_epoch_1.pt"
    # model_path = "/ceph/home/wangzl21/Projects/My_3/deep-motion-editing/diffusion_latent/experiments/reinforce_diffusion_baseline_onlydiff_withoutreturnnorm_gradnorm0.2_lr1e-5_seed0/ckpt/diffusion_epoch_1.pt"

    save_dir = config.save_dir
    model_path = config.model_path
    sample_fn, wavlm_model, vqvae_model, diffusion_model = main(save_dir, model_path)

    audio = config.audio_path       # "../dataset/ZEGGS/all_speech/067_Speech_2_x_1_0.npy"
    _, _, code, _, distance, _ = single_inference(audio, config, sample_fn, wavlm_model, vqvae_model, diffusion_model, time_step=False, w_grad=False, save_result=True)      # for debug, motion (1, 112, len)  code (len//2, 1)

    # generate_result('../dataset/ZEGGS/valid_speech/', '../dataset/Trinity/valid_speech/', config, sample_fn, wavlm_model, vqvae_model, diffusion_model, return_GT=False)

    # all_inference('../dataset/ZEGGS/all_speech/', '../dataset/Trinity/all_speech/', '../dataset/ZEGGS/valid_speech/', '../dataset/Trinity/valid_speech/',
    #               config, sample_fn, wavlm_model, vqvae_model, diffusion_model, return_GT=False)       # you may change return_GT to True to get GT code (y)

