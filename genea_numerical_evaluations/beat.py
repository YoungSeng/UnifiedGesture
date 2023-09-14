import librosa
import numpy as np
import pdb
import os
from  scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema

def calc_db(keypoints, name=''):
    keypoints = np.array(keypoints).reshape(-1, 33, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats, len(kinetic_vel)


def BA(music_beats, motion_beats):
    ba = 0
    for bb in music_beats:
        ba +=  np.exp(-np.min((motion_beats[0] - bb)**2) / 2 / 9)
    return (ba / len(music_beats))


def cal_beat_alien(audio_path, motion_path):
    ba_scores = []
    for audio in os.listdir(audio_path):
        audio_wav = np.load(os.path.join(audio_path, audio))[0]
        audio_beat_time = librosa.onset.onset_detect(y=audio_wav, sr=16000, units='frames')
        for item in os.listdir(motion_path):
            if audio.split('.')[0] in item.split('.')[0]:
                joint3d = np.load(os.path.join(motion_path, item))
                break
        gesture_beats, length = calc_db(joint3d)
        ba_scores.append(BA(audio_beat_time, gesture_beats))
    return np.mean(ba_scores), len(ba_scores)


if __name__ == '__main__':
    '''
    python beat.py
    '''
    audio_path = '/ceph/hdd/yangsc21/Python/My_3/10_npy/'
    motion_path = '/ceph/hdd/yangsc21/Python/My_3/wZEGGS_npy'
    print(cal_beat_alien(audio_path, motion_path))
