'''
pip install lmdb==1.3.0
pip install pyarrow==8.0.0  (-i http://pypi.douban.com/simple --trusted-host pypi.douban.com)
'''

import argparse
import os
import glob
import pdb
from pathlib import Path

# import librosa
import numpy as np
import lmdb
import pyarrow


def make_lmdb_latent_dataset(base_path):
    Trinity_latent_path = os.path.join(base_path, 'Trinity')
    ZEGGS_latent_path = os.path.join(base_path, 'ZEGGS')
    out_path = os.path.join(base_path, 'lmdb_latent_vel')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    map_size = 1024 * 20  # in MB
    map_size <<= 20  # in B
    db = [lmdb.open(os.path.join(out_path, 'lmdb_train'), map_size=map_size),
          lmdb.open(os.path.join(out_path, 'lmdb_test'), map_size=map_size)]

    # delete existing files
    for i in range(2):
        with db[i].begin(write=True) as txn:
            txn.drop(db[i].open_db())

    all_positions = []
    bvh_files = sorted(glob.glob(Trinity_latent_path + "/*_upper.npy") + glob.glob(ZEGGS_latent_path + "/*_upper.npy"))

    save_idx = 0
    for v_i, bvh_file in enumerate(bvh_files):
        path = os.path.split(bvh_file)[0]
        name = os.path.split(bvh_file)[1][:-4]
        print(name)

        # load skeletons and subtitles

        upper = np.load(bvh_file)[0].transpose()
        lower = np.load(os.path.join(path, name.replace('upper', 'lower') + '.npy'))[0].transpose()
        root = np.load(os.path.join(path, name.replace('upper', 'root') + '.npy'))[0].transpose()
        root_vel = np.load(os.path.join(path, name.replace('upper', 'root_vel') + '.npy'))[0].transpose()

        # global_position = np.load(os.path.join(path, name.replace('local', 'global') + '.npy'), allow_pickle=True)
        # position = global_position[0][0].transpose()
        # height = global_position[1]

        # print(upper.shape, lower.shape, root.shape)

        # load audio
        # audio_raw, audio_sr = librosa.load(os.path.join(audio_path, '{}.wav'.format(name)),
        #                                    mono=True, sr=16000, res_type='kaiser_fast')

        # process
        clips = [{'vid': name, 'clips': []},  # train
                 {'vid': name, 'clips': []}]  # validation

        # split
        if save_idx % 10 == 0:
            dataset_idx = 1  # validation
        else:
            dataset_idx = 0  # train

        # word preprocessing
        # word_list = []
        # for wi in range(len(subtitle)):
        #     word_s = float(subtitle[wi]['start_time'][:-1])
        #     word_e = float(subtitle[wi]['end_time'][:-1])
        #     word = subtitle[wi]['word']
        #
        #     word = normalize_string(word)
        #     if len(word) > 0:
        #         word_list.append([word, word_s, word_e])

        # save subtitles and skeletons
        # poses = np.asarray(poses)       # x dtype=np.float16
        clips[dataset_idx]['clips'].append(
            {
             'upper': upper,
             'lower': lower,
             'root': root,
             'root_vel': root_vel,
             })

        # write to db
        for i in range(2):
            with db[i].begin(write=True) as txn:
                if len(clips[i]['clips']) > 0:
                    k = '{:010}'.format(v_i).encode('ascii')
                    v = pyarrow.serialize(clips[i]).to_buffer()
                    txn.put(k, v)
        # all_positions.append(root)
        save_idx += 1

    # close db
    for i in range(2):
        db[i].sync()
        db[i].close()

    # calculate data mean
    # all_poses = np.vstack(all_positions)
    # pose_mean = np.mean(all_poses, axis=0, dtype=np.float64)
    # pose_std = np.std(all_poses, axis=0, dtype=np.float64)
    #
    # print('data mean/std')
    # print('data_mean:', str(["{:0.5f}".format(e) for e in pose_mean]).replace("'", ""))
    # print('data_std:', str(["{:0.5f}".format(e) for e in pose_std]).replace("'", ""))


if __name__ == '__main__':
    '''
    python ./datasets/latent_to_lmdb.py --base_path ./datasets/bvh2upper_lower_root
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=Path)
    args = parser.parse_args()

    make_lmdb_latent_dataset(args.base_path)
