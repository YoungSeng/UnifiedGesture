import pdb

import numpy as np
import os
import sys
sys.path.append('../utils')
import BVH as BVH
import Animation as Animation

def bvh2positon(source_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in os.listdir(source_path):
        print(file)
        if file.endswith('.bvh'):
            bvh_file = os.path.join(source_path, file)
            anim, names, frametime = BVH.load(bvh_file)
            glb = Animation.positions_global(anim)
            # root_position = glb[:, names.index('Hips')].copy()
            np.save(os.path.join(save_path, file[:-4] + '.npy'), glb)


if __name__ == '__main__':
    '''
    cd genea_numerical_evaluations
    python bvh2position.py
    '''
    source_path = "/ceph/hdd/yangsc21/Python/My_3/proposed"
    savepath = "/ceph/hdd/yangsc21/Python/My_3/proposed" + '_npy'
    bvh2positon(source_path, savepath)

