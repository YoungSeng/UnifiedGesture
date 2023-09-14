
import argparse
import os
import glob
import pdb
from pathlib import Path

# import librosa
import numpy as np
import lmdb
import pyarrow


style2onehot = {
'Happy':[1, 0, 0, 0, 0, 0, 0],
'Sad':[0, 1, 0, 0, 0, 0, 0],
'Neutral':[0, 0, 1, 0, 0, 0, 0],
'Old':[0, 0, 0, 1, 0, 0, 0],
'Angry':[0, 0, 0, 0, 1, 0, 0],
'Relaxed':[0, 0, 0, 0, 0, 1, 0],
'Still':[0, 0, 0, 0, 0, 0, 1]
}

def make_lmdb_latent_dataset(base_path):
    Trinity_path = os.path.join(base_path, 'Trinity')
    ZEGGS_path = os.path.join(base_path, 'ZEGGS')
    out_path = os.path.join(base_path, 'all_lmdb_aux')
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

    total_files = sorted(glob.glob(Trinity_path + "/all_gesture_aux" + "/*_upper.npy") + glob.glob(ZEGGS_path + "/all_gesture_aux" + "/*_upper.npy"))

    save_idx = 0
    for v_i, code_file in enumerate(total_files):
        path = os.path.split(code_file)[0]
        name = os.path.split(code_file)[1][:-4].replace('_upper', '')
        style = name.split('_')[1]

        print(name)
        if style in style2onehot:
            print(style)
            style = style2onehot[style]
        else:
            style = style2onehot['Neutral']

        # load
        upper_code = np.load(code_file)
        full_body = np.load(os.path.join(path, name + '_all_body.npy'))
        audio_wav = np.load(os.path.join(path.replace('all_gesture_aux', 'all_speech'), name) + '.npy')

        MIN_LEN = min(full_body.shape[0], int(audio_wav.shape[-1]*30/16000/4))
        full_body = full_body[:MIN_LEN]
        audio_wav = audio_wav[:int(MIN_LEN*16000*4/30)]

        # process
        print(upper_code.shape, full_body.shape, audio_wav.shape)

        # process
        clips = [{'vid': name, 'clips': []},  # train
                 {'vid': name, 'clips': []}]  # validation

        # split
        if save_idx % 10 == 0:
            dataset_idx = 1  # validation
        else:
            dataset_idx = 0  # train

        clips[dataset_idx]['clips'].append(
            {
              'audio_wav': audio_wav,
              'full_body': full_body,
              'upper_code': upper_code,
              'style': np.array(style),
             })

        # write to db
        for i in range(2):
            with db[i].begin(write=True) as txn:
                if len(clips[i]['clips']) > 0:
                    k = '{:010}'.format(v_i).encode('ascii')
                    v = pyarrow.serialize(clips[i]).to_buffer()
                    txn.put(k, v)
        save_idx += 1

    # close db
    for i in range(2):
        db[i].sync()
        db[i].close()


if __name__ == '__main__':
    '''
    python ./make_lmdb.py --base_path ./dataset/
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=Path)
    args = parser.parse_args()

    make_lmdb_latent_dataset(args.base_path)
