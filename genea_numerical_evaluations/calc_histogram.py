# -*- coding: utf-8 -*-
"""
Calculating statistics over the produced and ground truth gestures

@author: Naoshi Kaneko and Taras Kucherenko
"""

import argparse
import glob
import os
import pdb
import warnings

import matplotlib.pyplot as plt
import numpy as np



def compute_velocity(data, dim=3):
    """Compute velocity between adjacent frames

      Args:
          data:         array containing joint positions of gesture
          dim:          gesture dimensionality

      Returns:
          vel_norms:    velocities of each joint between each adjacent frame
    """

    # Flatten the array of 3d coords
    coords = data.reshape(data.shape[0], -1)

    # First derivative of position is velocity
    vels = np.diff(coords, n=1, axis=0)

    num_vels = vels.shape[0]
    num_joints = vels.shape[1] // dim

    vel_norms = np.zeros((num_vels, num_joints))

    # calculate vector norms
    for i in range(num_vels):
        for j in range(num_joints):
            x1 = j * dim + 0
            x2 = j * dim + dim
            vel_norms[i, j] = np.linalg.norm(vels[i, x1:x2])

    # multiply on 20 to compensate for the time-frame being  0.05s
    return vel_norms * 20


def compute_acceleration(data, dim=3):
    """Compute acceleration between adjacent frames

      Args:
          data:         array containing joint positions of gesture
          dim:          gesture dimensionality

      Returns:
          acc_norms:    accelerations of each joint between each adjacent frame
    """

    # Second derivative of position is acceleration
    accs = np.diff(data, n=2, axis=0)

    num_accs = accs.shape[0]
    num_joints = accs.shape[1] // dim

    acc_norms = np.zeros((num_accs, num_joints))

    # calculate vector norms
    for i in range(num_accs):
        for j in range(num_joints):
            x1 = j * dim + 0
            x2 = j * dim + dim
            acc_norms[i, j] = np.linalg.norm(accs[i, x1:x2])

    # multiply on 20 to compensate for the time-frame being  0.05s
    return acc_norms * 20 * 20


def save_result(lines, out_dir, width, measure):
    """Write computed histogram to CSV

      Args:
          lines:        list of strings to be written
          out_dir:      output directory
          width:        bin width of the histogram
          measure:      used measure for histogram calculation
    """

    # Make output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    hist_type = measure[:3]  # 'vel' or 'acc'
    filename = 'hmd_{}_{}.csv'.format(hist_type, width)
    outname = os.path.join(out_dir, filename)

    with open(outname, 'w') as out_file:
        out_file.writelines(lines)


def make_histogram(cond_name, measure, coord_dir_, out_dir, visualize=True, width=1):
    """
    Calculate frequencies histogram for a given measure
    Args:
        cond_name:   name of the folder to consider
        measure:     measure to be used
        coord_dir:   folder where all the data for the current model is stored
        out_dir:     folder where the results should be stored
        visualize:   flag if we want to visualize the result
        width:       width of the histogram bins

    Returns:
        Saved in the "result" folder

    """

    # coord_dir = "<..your path/GENEA/genea_challenge_2022/dataset/v1_18_1/val/>"
    # cond_name = 'npy'

    cond_name = coord_dir_.split('/')[-2]
    coord_dir = '/'.join(coord_dir_.split('/')[:-2])
    predicted_dir = os.path.join(coord_dir, cond_name)

    measures = {
        'velocity': compute_velocity,
        'acceleration': compute_acceleration,
    }

    # Check if error measure was correct
    if measure not in measures:
        raise ValueError('Unknown measure: \'{}\'. Choose from {}'
                         ''.format(measure, list(measures.keys())))

    predicted_files = sorted(glob.glob(os.path.join(predicted_dir, '*.npy')))

    predicted_out_lines = [','.join([''] + ['Total']) + '\n']

    predicted_distances = []
    for predicted_file in predicted_files:
        # read the values and remove hips which are fixed
        predicted_coords = np.load(predicted_file)[:, 2:]

        predicted_distance = measures[measure](predicted_coords)  # [:, selected_joints]

        predicted_distances.append(predicted_distance)

    predicted_distances = np.concatenate(predicted_distances)

    # Compute histogram for each joint
    bins = np.arange(0, 59 + width, width)
    num_joints = predicted_distances.shape[1]
    predicted_hists = []
    for i in range(num_joints):
        predicted_hist, _ = np.histogram(predicted_distances[:, i], bins=bins)

        predicted_hists.append(predicted_hist)

    # Sum over all joints
    predicted_total = np.sum(predicted_hists, axis=0)

    # Append total number of bin counts to the last
    predicted_hists = np.stack(predicted_hists + [predicted_total], axis=1)

    num_bins = bins.size - 1
    for i in range(num_bins):
        predicted_line = str(bins[i])
        for j in range(num_joints + 1):
            predicted_line += ',' + str(predicted_hists[i, j])
        predicted_line += '\n'
        predicted_out_lines.append(predicted_line)

    out_dir = os.path.join('/ceph/hdd/yangsc21/Python/My_3/', 'result')
    predicted_out_dir = os.path.join(out_dir, cond_name)

    # predicted_out_dir = out_dir

    if visualize:
        import matplotlib
        matplotlib.use('AGG')

        sum = np.sum(predicted_total)
        actual_freq = predicted_total / sum
        plt.plot(bins[:-1], actual_freq, label=cond_name)
        plt.legend()
        plt.xlabel('Velocity (cm/s)')
        plt.ylabel('Frequency')
        plt.title('Frequencies of Moving Distance ({})'.format(measure))
        plt.tight_layout()
        plt.savefig('2.jpg')
        plt.show()

    save_result(predicted_out_lines, predicted_out_dir, width, measure)

    print('HMD ({}):'.format(measure))
    print('bins: {}'.format(bins))
    print('population: {}'.format(predicted_total))

def main():
    '''
    python calc_histogram.py
    python calc_histogram.py -p your_prediction_dir -m velocity # You can change the bin width of the histogram
    '''
    parser = argparse.ArgumentParser(
        description='Calculate histograms of moving distances')
    parser.add_argument('--coords_dir', '-p', default='data',
                        help='Predicted gestures directory')
    parser.add_argument('--width', '-w', type=float, default=1,
                        help='Bin width of the histogram')
    parser.add_argument('--measure', '-m', default='acceleration',      # acceleration/velocity
                        help='Measure to calculate (velocity or acceleration)')
    parser.add_argument('--select', '-s', nargs='+',
                        help='Joint subset to compute (if omitted, use all)')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Visualize histograms')
    parser.add_argument('--out_dir', default='resulting',
                        help='Directory to output the result')
    args = parser.parse_args()

    # Make sure that data is stored in the correct folder
    # if not os.listdir(args.coords_dir):
    #     print("--coords_dir argument is wrong. there is no data at the folder '", args.coords_dir, "'")
    #     exit(-1)
    #
    # for cond_name in os.listdir(args.coords_dir):
    #     print("\nConsider", cond_name)
    #     make_histogram(cond_name, args.measure, args.coords_dir, args.out_dir, args.visualize, args.width)

    make_histogram(None, args.measure, args.coords_dir, args.out_dir, True, args.width)

    print('\nMore detailed result was writen to the files in the ',  args.out_dir, ' folder ')
    print('')

if __name__ == '__main__':
    main()
    '''
    HMD (velocity):
bins: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
 48 49 50 51 52 53 54 55 56 57 58 59]
population: [625273 320410 205760 155278 124341 102980  86938  70758  65529  53843
  48623  45588  35938  37620  28903  31020  25701  25250  21911  20780
  19008  18130  16544  16517  13187  14551  11563  12837  10702  11117
   9425   9779   8670   9162   7326   8170   6646   7215   6046   6602
   5633   6001   5100   5238   4769   5030   4246   4717   3826   4341
   3946   3797   3502   3520   3311   3234   3046   3110   2797]
    
    HMD (acceleration):
bins: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
 48 49 50 51 52 53 54 55 56 57 58 59]
population: [227618  45027  37958  34907  25682  19830  16445  54049  27785  13353
  10919  10299  16592  20684  10678  19987  13479   7893   7355  10320
  13200   7920   7246  14707  16362  13671   8736   7863   6880   7136
   7503  26425   7367  17333   8120   9997  11934  31960  19732 123117
  38035  17649  13405  10394   9272   8852  18737  16649   8916  10373
  18598   9610   8088   7064  11053  11297  24689   6829   6914]

    '''