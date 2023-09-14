import os
import glob
import pdb

import numpy as np


def process_code(source_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    upper_files = sorted(glob.glob(source_path + "/code_upper_*.npy"))
    for item in upper_files:
        name = os.path.split(item)[1].replace('code_upper_', '')
        print(name)
        upper = np.load(item)
        lower = np.load(item.replace('upper', 'lower'))
        # root = np.load(item.replace('upper', 'root'))
        # print(upper.shape, lower.shape, root.shape)
        # latent_code = np.concatenate([upper, lower, root], axis=1)

        print(upper.shape, lower.shape)
        latent_code = np.concatenate([upper, lower], axis=1)
        latent_code = latent_code
        np.save(os.path.join(save_path, name), latent_code)


def process_motion(source_code_path, source_motion_path, source_aux_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    root_motions = sorted(glob.glob(source_motion_path + "/*_root.npy"))
    for item in root_motions:
        name = os.path.split(item)[1].replace('_root.npy', '')
        print(name)
        root_motion = np.load(item)
        lower_motion = np.load(os.path.join(source_motion_path, name + '_lower' + '.npy'))
        upper_motion = np.load(os.path.join(source_motion_path, name + '_upper' + '.npy'))
        all_body_motion = np.concatenate([lower_motion, upper_motion, root_motion], axis=1)
        all_body_motion = all_body_motion[0].transpose(1, 0)      # (len, 112)

        all_body_vel = all_body_motion[:-1] - all_body_motion[1:]
        all_body_acc = all_body_vel[:-1] - all_body_vel[1:]
        clip_skeleton_vel = np.pad(all_body_vel, ((1, 0), (0, 0)), 'constant')
        clip_skeleton_acc = np.pad(all_body_acc, ((2, 0), (0, 0)), 'constant')

        aux = np.load(os.path.join(source_aux_path, name + '.bvh.npy'))
        aux = aux[::4][:aux.shape[0]//4]

        all_body = np.concatenate([all_body_motion, clip_skeleton_vel, clip_skeleton_acc, aux], axis=1)

        upper_code = np.load(os.path.join(source_code_path, name + '.npy'))
        upper_code = upper_code[:, 0]      # (2694,)

        np.save(os.path.join(save_path, name + '_upper' + '.npy'), upper_code)
        np.save(os.path.join(save_path, name + '_all_body' + '.npy'), all_body)


if __name__ == '__main__':
    '''
    python process_code.py
    '''
    Trinity_VQVAE_path = './dataset/Trinity/VQVAE_result/Trinity/'
    ZEGGS_VQVAE_path = './dataset/ZEGGS/VQVAE_result/ZEGGS/'

    Trinity_save_code_path = './dataset/Trinity/latent_code_2/'
    ZEGGS_save_code_path = './dataset/ZEGGS/latent_code_2/'

    # process_code(Trinity_VQVAE_path, Trinity_save_code_path)
    # process_code(ZEGGS_VQVAE_path, ZEGGS_save_code_path)

    # Trinity_source_motion_path = './retargeting/datasets/bvh2upper_lower_root/Trinity/'
    # ZEGGS_source_motion_path = './retargeting/datasets/bvh2upper_lower_root/ZEGGS/'
    # Trinity_save_path = './dataset/Trinity/gesture/'
    # ZEGGS_save_path = './dataset/ZEGGS/gesture/'
    # process_motion(Trinity_save_code_path, Trinity_source_motion_path, Trinity_save_path)
    # process_motion(ZEGGS_save_code_path, ZEGGS_source_motion_path, ZEGGS_save_path)

    Trinity_source_motion_path = './retargeting/datasets/bvh2upper_lower_root/Trinity/'
    ZEGGS_source_motion_path = './retargeting/datasets/bvh2upper_lower_root/ZEGGS/'
    Trinity_source_aux_path = "./retargeting/datasets/Mixamo_new_2/Trinity_aux/"
    ZEGGS_source_aux_path = "./retargeting/datasets/Mixamo_new_2/ZEGGS_aux/"
    Trinity_save_path = './dataset/Trinity/all_gesture_aux/'
    ZEGGS_save_path = './dataset/ZEGGS/all_gesture_aux/'
    process_motion(Trinity_save_code_path, Trinity_source_motion_path, Trinity_source_aux_path, Trinity_save_path)
    process_motion(ZEGGS_save_code_path, ZEGGS_source_motion_path, ZEGGS_source_aux_path, ZEGGS_save_path)
