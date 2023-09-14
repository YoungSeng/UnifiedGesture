# -*- coding: utf-8 -*-
"""
Calculating CCA (Canonical Correlation Analysis) over the produced and ground truth gestures

@author: Taras Kucherenko
"""

import argparse
import glob
import os
import numpy as np

from cca import calculate_CCA_score, find_CCA_scaling_vectors


def shorten(arr_one, arr_two):
    """
    Make sure that two arrays have the same length
    Args:
        arr_one:  array one
        arr_two:  array two

    Returns:
        shortened versions of both arrays

    """
    min_len = min(arr_one.shape[0], arr_two.shape[0])
    arr_one = arr_one[:min_len]
    arr_two = arr_two[:min_len]

    return arr_one, arr_two


def save_result(lines, out_dir):
    """Write computed measure to CSV

      Args:
          lines:        list of strings to be written
          out_dir:      output directory
          measure:      used measure
    """

    # Make output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    outname = os.path.join(out_dir, 'cca.csv')

    with open(outname, 'w') as out_file:
        out_file.writelines(lines)


# def evaluate_folder(cond_name, coords_dir):
def evaluate_folder(cond_dir, gt_dir):
    """
    Calculate numerical measure for the coordinates in the given folder
    Args:
        cond_name:   name of the condition / folder to evaluate
        coords_dir:  folder where all the data for the current model is stored

    Returns:
        nothing, prints out the metrics results

    """

    # cond_dir = os.path.join(coords_dir, cond_name)
    # gt_dir = os.path.join(coords_dir, "GT")

    generated_files = sorted(glob.glob(os.path.join(cond_dir, '*.npy')))

    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.npy')))

    # First - find the CCA scaling vectors using all the data for the given model
    all_predicted_frames = []
    all_ground_tr_frames = []
    for predicted_file, gt_file in zip(generated_files, gt_files):

        # read and flatten the predicted values
        predicted_coords = np.load(predicted_file)
        predicted_coords = np.reshape(predicted_coords, (predicted_coords.shape[0], -1))

        # read and flatten the ground truth values
        original_coords = np.load(gt_file)
        original_coords = np.reshape(original_coords, (original_coords.shape[0], -1))

        # make sure sequences have the same length
        predicted_coords, original_coords = shorten(predicted_coords, original_coords)

        if len(all_predicted_frames) != 0:
            all_predicted_frames = np.concatenate((all_predicted_frames, predicted_coords), axis=0)
            all_ground_tr_frames = np.concatenate((all_ground_tr_frames, original_coords), axis=0)
        else:
            all_predicted_frames = predicted_coords
            all_ground_tr_frames = original_coords

    # find CCA models
    cca_model = find_CCA_scaling_vectors(all_predicted_frames, all_ground_tr_frames)

    # calculate Global CCA value
    global_cca_value = calculate_CCA_score(all_predicted_frames, all_ground_tr_frames, cca_model)
    print('Global CCA value: {:.5f}'.format(global_cca_value))


    # calculate CCA value for each sequence
    predicted_out_lines = [','.join(['file']) + '\n']

    predicted_errors = []
    for predicted_file, gt_file in zip(generated_files, gt_files):
        # read and flatten the predicted values
        predicted_coords = np.load(predicted_file)
        predicted_coords = np.reshape(predicted_coords, (predicted_coords.shape[0], -1))

        # read and flatten the ground truth values
        original_coords = np.load(gt_file)
        original_coords = np.reshape(original_coords, (original_coords.shape[0], -1))

        # make sure sequences have the same length
        predicted_coords, original_coords = shorten(predicted_coords, original_coords)

        # calculate CCA value
        current_cca_value = calculate_CCA_score(original_coords, predicted_coords, cca_model)

        predicted_errors.append(current_cca_value)

        basename = os.path.basename(predicted_file)
        predicted_line = basename

        predicted_line += ',' + str(current_cca_value) + '\n'

        predicted_out_lines.append(predicted_line)

    predicted_average_line = 'Average'
    error_avgs = np.mean(predicted_errors, axis=0)
    error_stds = np.std(predicted_errors, axis=0)

    predicted_average_line += ',' + str(error_avgs)

    predicted_out_lines.append(predicted_average_line)

    # predicted_out_dir = os.path.join("result", cond_name)
    #
    # save_result(predicted_out_lines, predicted_out_dir)
    #
    # print('{:s}: {:.2f} +- {:.2F}'.format(cond_name, np.mean(predicted_errors), error_stds))
    print('{:.2f} +- {:.2F}'.format(np.mean(predicted_errors), error_stds))


if __name__ == '__main__':
    '''
    python calc_cca.py --cond_dir "<..your path/GENEA/genea_challenge_2022/baselines/Tri/output/infer_sample/output_2_new_wavlm/npy/>" --gt_dir "<..your path/GENEA/genea_challenge_2022/dataset/v1_18_1/val/npy/>"
    '''
    parser = argparse.ArgumentParser(
        description='Calculate prediction errors')
    # parser.add_argument('--coords_dir', '-c', default='data',
    #                     help='Predicted gesture directory')
    parser.add_argument('--cond_dir', '-c', default='data',
                        help='Predicted gesture directory')
    parser.add_argument('--gt_dir', '-g', default='data',
                        help='Predicted gesture directory')
    args = parser.parse_args()

    # Make sure that data is stored in the correct folder
    # if not os.listdir(args.coords_dir):
    #     print("--coords_dir argument is wrong. there is no data at the folder '", args.coords_dir, "'")
    #     exit(-1)

    print('CCA:')

    # for cond_name in os.listdir(args.coords_dir):
    #     evaluate_folder(cond_name, args.coords_dir)

    evaluate_folder(args.cond_dir, args.gt_dir)

    '''
    CCA 
    output_2_new_wavlm  0.97 +- 0.01
    '''
