# -*- coding: utf-8 -*-
"""
Calculating average jerk over the produced and ground truth gestures

@author: Taras Kucherenko and Naoshi Kaneko
"""

import argparse
import glob
import os
import pdb
import warnings

import numpy as np


def compute_jerks(data, dim=3):
    """Compute jerk between adjacent frames

      Args:
          data:         array containing joint positions of gesture
          dim:          gesture dimensionality

      Returns:
          np.ndarray:   jerks of each joint averaged over all frames
    """

    # Third derivative of position is jerk
    jerks = np.diff(data, n=3, axis=0)

    num_jerks = jerks.shape[0]
    num_joints = jerks.shape[1] // dim

    jerk_norms = np.zeros((num_jerks, num_joints))

    for i in range(num_jerks):
        for j in range(num_joints):
            x1 = j * dim + 0
            x2 = j * dim + dim
            jerk_norms[i, j] = np.linalg.norm(jerks[i, x1:x2])

    average = np.mean(jerk_norms, axis=0)

    # Take into account that frame rate was 20 fps
    scaled_av = average * 20 * 20 * 20

    return scaled_av


def compute_acceleration(data, dim=3):
    """Compute acceleration between adjacent frames

      Args:
          data:         array containing joint positions of gesture
          dim:          gesture dimensionality

      Returns:
          np.ndarray:   accelerations of each joint averaged over all frames
    """

    # Second derivative of position is acceleration
    accs = np.diff(data, n=2, axis=0)

    num_accs = accs.shape[0]
    num_joints = accs.shape[1] // dim

    acc_norms = np.zeros((num_accs, num_joints))

    for i in range(num_accs):
        for j in range(num_joints):
            x1 = j * dim + 0
            x2 = j * dim + dim
            acc_norms[i, j] = np.linalg.norm(accs[i, x1:x2])

    average = np.mean(acc_norms, axis=0)

    # Take into account that frame rate was 20 fps
    scaled_av = average * 20 * 20

    return scaled_av


def save_result(lines, out_dir, measure):
    """Write computed measure to CSV

      Args:
          lines:        list of strings to be written
          out_dir:      output directory
          measure:      used measure
    """

    # Make output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if measure == "jerk":
        outname = os.path.join(out_dir, 'aj.csv')
    elif measure == "acceleration":
        outname = os.path.join(out_dir, 'aa.csv')

    with open(outname, 'w') as out_file:
        out_file.writelines(lines)


def evaluate_folder(cond_name, coord_dir, measure):
    """
    Calculate numerical measure for the coordinates in the given folder
    Args:
        cond_name:   name of the condition / folder to evaluate
        coord_dir:   folder where all the data for the current model is stored
        measure:     measure to be used

    Returns:
        nothing, prints out the metrics results

    """

    # cond_dir = os.path.join(coord_dir, cond_name)

    cond_dir = coord_dir

    cond_files = sorted(glob.glob(os.path.join(cond_dir, '*.npy')))

    # define possible measures
    measures = {
        'jerk': compute_jerks,
        'acceleration': compute_acceleration,
    }

    # Check if error measure was correct
    if measure not in measures:
        raise ValueError('Unknown measure: \'{}\'. Choose from {}'
                         ''.format(measure, list(measures.keys())))

    predicted_out_lines = [','.join(['file']) + '\n']

    all_motion_stats = []
    for predicted_file in cond_files:
        predicted_coords = np.load(predicted_file)

        # flatten the values
        predicted_coords = np.reshape(predicted_coords, (predicted_coords.shape[0], -1))

        current_motion_stats = measures[measure](predicted_coords)

        all_motion_stats.append(current_motion_stats)

        basename = os.path.basename(predicted_file)
        predicted_line = basename

        for ov in current_motion_stats:
            predicted_line += ',' + str(ov)

        predicted_line += '\n'

        predicted_out_lines.append(predicted_line)

    predicted_average_line = 'Average'

    avgs_for_each_joint = np.mean(all_motion_stats, axis=0)

    avgs_for_each_sequence = np.mean(all_motion_stats, axis=1)

    stds_over_sequences = np.std(avgs_for_each_sequence)
    mean_over_sequences = np.mean(avgs_for_each_sequence)

    for oa in avgs_for_each_joint:
        predicted_average_line += ',' + str(oa)

    predicted_out_lines.append(predicted_average_line)

    # predicted_out_dir = os.path.join("result", cond_name)

    # save_result(predicted_out_lines, predicted_out_dir, measure)

    print('{:.2f} +- {:.2F}'.format(mean_over_sequences, stds_over_sequences))


if __name__ == '__main__':
    '''
    python calk_jerk_or_acceleration.py -m jerk --coords_dir "<..your path/GENEA/genea_challenge_2022/baselines/Tri/output/infer_sample/output_2_new_wavlm/npy/>"
    '''
    parser = argparse.ArgumentParser(
        description='Calculate prediction errors')
    parser.add_argument('--coords_dir', '-c', default='data',
                        help='Predicted gesture directory')
    parser.add_argument('--measure', '-m', default='acceleration',
                        help='Measure to calculate (jerk or acceleration)')
    args = parser.parse_args()

    # Make sure that data is stored in the correct folder
    if not os.listdir(args.coords_dir):
        print("--coords_dir argument is wrong. there is no data at the folder '", args.coords_dir, "'")
        exit(-1)

    if args.measure == 'jerk':
        print('AJ:')
    elif args.measure == 'acceleration':
        print('AA:')

    # for cond_name in os.listdir(args.coords_dir):
    #     if cond_name == "GT":
    #         continue
    evaluate_folder(None, args.coords_dir, args.measure)

    print('More detailed result was writen to the files in the "result" folder ')
    print('')

    '''
    GT  AJ  6184.28 +- 2235.72 cm/s3    AA  198.10 +- 69.12 cm/s2
    output_2_new_wavlm  AJ 550.74 +- 297.20      AA  38.76 +- 14.76
    
    '''
