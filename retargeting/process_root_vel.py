import os
import glob
import pdb

import numpy as np

def extract_vel(path):
    root_files = sorted(glob.glob(path + "/*_root.npy"))
    for item in root_files:
        print(item)
        root = np.load(item)
        root_vel = root[..., 1:] - root[..., :-1]
        root_vel = np.pad(root_vel, [(0, 0), (0, 0), (1, 0)], mode='constant')
        np.save(item.replace('_root.npy', '_root_vel.npy'), root_vel)


if __name__ == '__main__':
    '''
    python process_root_vel.py
    '''
    Trinity_path = '../retargeting/datasets/Trinity_ZEGGS/bvh2upper_lower_root/Trinity/'
    ZEGGS_path = '../retargeting/datasets/Trinity_ZEGGS/bvh2upper_lower_root/ZEGGS/'

    extract_vel(Trinity_path)
    extract_vel(ZEGGS_path)
