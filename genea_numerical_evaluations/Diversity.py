import pdb

import numpy as np
import os

def extract_kinetic_features(positions):
    assert len(positions.shape) == 3  # (seq_len, n_joints, 3)
    features = KineticFeatures(positions)
    kinetic_feature_vector = []
    for i in range(positions.shape[1]):
        feature_vector = np.hstack(
            [
                features.average_kinetic_energy_horizontal(i),
                features.average_kinetic_energy_vertical(i),
                features.average_energy_expenditure(i),
            ]
        )
        kinetic_feature_vector.extend(feature_vector)
    kinetic_feature_vector = np.array(kinetic_feature_vector, dtype=np.float32)
    return kinetic_feature_vector


def calc_average_velocity_vertical(
    positions, i, joint_idx, sliding_window, frame_time, up_vec
):
    current_window = 0
    average_velocity = np.zeros(len(positions[0][joint_idx]))
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j >= len(positions):
            continue
        average_velocity += (
            positions[i + j][joint_idx] - positions[i + j - 1][joint_idx]
        )
        current_window += 1
    if up_vec == "y":
        average_velocity = np.array([average_velocity[1]]) / (
            current_window * frame_time
        )
    elif up_vec == "z":
        average_velocity = np.array([average_velocity[2]]) / (
            current_window * frame_time
        )
    else:
        raise NotImplementedError
    return np.linalg.norm(average_velocity)


def calc_average_acceleration(
    positions, i, joint_idx, sliding_window, frame_time
):
    current_window = 0
    average_acceleration = np.zeros(len(positions[0][joint_idx]))
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j + 1 >= len(positions):
            continue
        v2 = (
            positions[i + j + 1][joint_idx] - positions[i + j][joint_idx]
        ) / frame_time
        v1 = (
            positions[i + j][joint_idx]
            - positions[i + j - 1][joint_idx] / frame_time
        )
        average_acceleration += (v2 - v1) / frame_time
        current_window += 1
    return np.linalg.norm(average_acceleration / current_window)


def calc_average_velocity_horizontal(
    positions, i, joint_idx, sliding_window, frame_time, up_vec="z"
):
    current_window = 0
    average_velocity = np.zeros(len(positions[0][joint_idx]))
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j >= len(positions):
            continue
        average_velocity += (
            positions[i + j][joint_idx] - positions[i + j - 1][joint_idx]
        )
        current_window += 1
    if up_vec == "y":
        average_velocity = np.array(
            [average_velocity[0], average_velocity[2]]
        ) / (current_window * frame_time)
    elif up_vec == "z":
        average_velocity = np.array(
            [average_velocity[0], average_velocity[1]]
        ) / (current_window * frame_time)
    else:
        raise NotImplementedError
    return np.linalg.norm(average_velocity)


def calc_average_velocity(positions, i, joint_idx, sliding_window, frame_time):
    current_window = 0
    average_velocity = np.zeros(len(positions[0][joint_idx]))
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j >= len(positions):
            continue
        average_velocity += (
            positions[i + j][joint_idx] - positions[i + j - 1][joint_idx]
        )
        current_window += 1
    return np.linalg.norm(average_velocity / (current_window * frame_time))


class KineticFeatures:
    def __init__(
        self, positions, frame_time=1./30, up_vec="y", sliding_window=2
    ):
        self.positions = positions
        self.frame_time = frame_time
        self.up_vec = up_vec
        self.sliding_window = sliding_window

    def average_kinetic_energy(self, joint):
        average_kinetic_energy = 0
        for i in range(1, len(self.positions)):
            average_velocity = calc_average_velocity(
                self.positions, i, joint, self.sliding_window, self.frame_time
            )
            average_kinetic_energy += average_velocity ** 2
        average_kinetic_energy = average_kinetic_energy / (
            len(self.positions) - 1.0
        )
        return average_kinetic_energy

    def average_kinetic_energy_horizontal(self, joint):
        val = 0
        for i in range(1, len(self.positions)):
            average_velocity = calc_average_velocity_horizontal(
                self.positions,
                i,
                joint,
                self.sliding_window,
                self.frame_time,
                self.up_vec,
            )
            val += average_velocity ** 2
        val = val / (len(self.positions) - 1.0)
        return val

    def average_kinetic_energy_vertical(self, joint):
        val = 0
        for i in range(1, len(self.positions)):
            average_velocity = calc_average_velocity_vertical(
                self.positions,
                i,
                joint,
                self.sliding_window,
                self.frame_time,
                self.up_vec,
            )
            val += average_velocity ** 2
        val = val / (len(self.positions) - 1.0)
        return val

    def average_energy_expenditure(self, joint):
        val = 0.0
        for i in range(1, len(self.positions)):
            val += calc_average_acceleration(
                self.positions, i, joint, self.sliding_window, self.frame_time
            )
        val = val / (len(self.positions) - 1.0)
        return val


def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    print(n)
    for i in range(n):
        print(i, end='\r')
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist


def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)

    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)


def calc_diversity(feats):
    feat_array = np.array(feats)
    n, c = feat_array.shape
    diff = np.array([feat_array] * n) - feat_array.reshape(n, 1, c)
    return np.sqrt(np.sum(diff**2, axis=2)).sum() / n / (n-1)


def cal_div(GT_path, source_path):
    pred_features = []
    for item in os.listdir(source_path):
        if item.endswith('.npy'):
            feat_source = np.load(os.path.join(source_path, item))
            pred_features.append(feat_source)
    GT_features = []
    for item in os.listdir(GT_path):
        if item.endswith('.npy'):
            GT_feat_source = np.load(os.path.join(GT_path, item))
            GT_features.append(GT_feat_source)

    pred_features_k = np.stack(pred_features)
    GT_features_k = np.stack(GT_features)
    GT_features_k, pred_features_k = normalize(GT_features_k, pred_features_k)
    # return calc_diversity(pred_features_k)
    return calculate_avg_distance(pred_features_k)


def extract_feat(source_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for item in os.listdir(source_path):
        if item.endswith('.npy'):
            joint3d = np.load(os.path.join(source_path, item)).reshape(-1, 33 * 3)
            roott = joint3d[:1, :3]
            joint3d = joint3d - np.tile(roott, (1, 33))
            feat = extract_kinetic_features(joint3d.reshape(-1, 33, 3))
            print(item, feat.shape)
            np.save(os.path.join(save_path, item), feat)


if __name__ == '__main__':
    '''
    python Diversity.py
    '''
    source_path = '/ceph/hdd/yangsc21/Python/My_3/worl_npy'
    GT_path = '/ceph/hdd/yangsc21/Python/My_3/GT_Gesture_npy'
    # extract_feat(source_path, source_path + '_feat')
    print(cal_div(GT_path + '_feat', source_path + '_feat'))
