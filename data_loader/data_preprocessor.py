""" create data samples """
import lmdb
import math
import numpy as np
import pyarrow


class DataPreprocessor:
    def __init__(self, clip_lmdb_dir, out_lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps):
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps

        self.src_lmdb_env = lmdb.open(clip_lmdb_dir, readonly=True, lock=False)
        with self.src_lmdb_env.begin() as txn:
            self.n_videos = txn.stat()['entries']

        self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * 75)
        self.lower_sample_length = int(self.n_poses * 2)

        # create db for samples
        map_size = 1024 * 1024 * 9  # in TB
        map_size <<= 20  # in B
        self.dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=map_size)
        self.n_out_samples = 0

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
        clip_speech = clip['speech']
        clip_upper = clip['upper']
        clip_lower = clip['lower']

        # divide
        aux_info = []
        sample_speech_list = []
        sample_gesture_list = []
        sample_lower_list = []

        # MINLEN = min(len(clip_skeleton), int(len(clip_audio_raw) * 60 / 16000), len(clip_codes_raw) * 8)
        MINLEN = len(clip_upper)

        num_subdivision = math.floor(
            (MINLEN - self.n_poses)
            / self.subdivision_stride) + 1  # floor((K - (N+M)) / S) + 1

        for i in range(num_subdivision):
            start_idx = i * self.subdivision_stride
            fin_idx = start_idx + self.n_poses
            sample_gesture = clip_upper[start_idx:fin_idx]

            subdivision_start_time = start_idx / self.skeleton_resampling_fps
            subdivision_end_time = fin_idx / self.skeleton_resampling_fps

            # raw audio
            audio_start = math.floor(start_idx / len(clip_upper) * len(clip_speech))
            audio_end = audio_start + self.audio_sample_length
            sample_audio = clip_speech[audio_start:audio_end]
            # raw lower
            lower_start = math.floor(start_idx / len(clip_upper) * len(clip_lower))
            lower_end = lower_start + self.lower_sample_length
            sample_lower = clip_lower[lower_start:lower_end]

            motion_info = {'vid': vid,
                           'start_frame_no': start_idx,
                           'end_frame_no': fin_idx,
                           'start_time': subdivision_start_time,
                           'end_time': subdivision_end_time}

            sample_gesture_list.append(sample_gesture)
            sample_speech_list.append(sample_audio)
            sample_lower_list.append(sample_lower)

            aux_info.append(motion_info)

        if len(sample_gesture_list) > 0:
            with self.dst_lmdb_env.begin(write=True) as txn:
                for gesture, lower, speech, aux in zip(sample_gesture_list, sample_lower_list, sample_speech_list, aux_info):
                    speech = np.asarray(speech)
                    lower = np.asarray(lower)
                    gesture = np.asarray(gesture)

                    # save
                    k = '{:010}'.format(self.n_out_samples).encode('ascii')
                    v = [speech, gesture, lower, aux]
                    v = pyarrow.serialize(v).to_buffer()
                    txn.put(k, v)
                    self.n_out_samples += 1
