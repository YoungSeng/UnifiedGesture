# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:30:40 2020

@authors: Naoshi Kaneko, Taras Kucherenko
"""

import argparse
import glob
import os
import pdb
import re

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


def normalize(hist):
    return hist / np.sum(hist)


def hellinger(hist1, hist2):
    """Compute Hellinger distance between two histograms

      Args:
          hist1:        first histogram
          hist2:        second histogram of the same size as hist1

      Returns:
          float:        Hellinger distance between hist1 and hist2
    """

    return np.sqrt(1.0 - np.sum(np.sqrt(normalize(hist1) * normalize(hist2))))


# https://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort  # NOQA
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]


def natural_sort(l, key=natural_sort_key):
    return sorted(l, key=key)


def main():
    '''
    python calc_histogram.py --original result/GT --predicted result/ --file hmd_vel_1.csv
    :return:
    '''
    parser = argparse.ArgumentParser(
        description='Calculate histograms of moving distances')
    parser.add_argument('--original', default="/ceph/hdd/yangsc21/Python/My_3/result/GT_Gesture_npy/",
                        help='Original gesture directory')
    parser.add_argument('--predicted', '-p', default="/ceph/hdd/yangsc21/Python/My_3/result/Audio2Gesture-20fps_npy/",
                        help='Predicted gesture directory')
    parser.add_argument('--file', '-f', default='hmd_vel_0.05.csv',
                        help='File name to load')
    parser.add_argument('--select', '-s', nargs='+',
                        help='Joint subset to compute (if omitted, use all)')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Visualize histograms')
    parser.add_argument('--out', '-o', default='results',
                        help='Directory to output the result')
    args = parser.parse_args()


    def get_directories(directory):
        return sorted(filter(lambda x: os.path.isdir(x), glob.glob(directory)))

    def get_histograms(data_dir, hist_file):

        # Read original gesture's distribution
        hist_path = os.path.join(data_dir, hist_file)
        original_val = pd.read_csv(hist_path, header=None, skiprows=1)
        original_array = np.array(original_val)

        # Calculate histograms for wrists and normalize it
        actual_hist = (original_array[:, -2] + original_array[:, -5]) / original_array[:, -1]

        return actual_hist

    original_hist = get_histograms(args.original, args.file)

    # List of predicted gesture directories
    # predicted_dirs = get_directories(os.path.join(args.predicted, '*'))
    predicted_dirs = get_directories(args.predicted)

    results = {os.path.basename(d): None for d in predicted_dirs}

    # Iterate over the list of directories
    for predicted_dir in predicted_dirs:
        # Does this directory have a target file?
        try:
            predicted_hist = get_histograms(predicted_dir, args.file)
        except FileNotFoundError:
            # Are there any subdirectories which have integer names?
            sub_dirs = sorted(
                filter(lambda x: os.path.basename(x).isdecimal(),
                get_directories(os.path.join(predicted_dir, '*'))))

            # If no, raise an exception
            if not sub_dirs:
                raise FileNotFoundError(
                    'There is neither ' + args.file
                    + ' nor subdirectories in ' + predicted_dir)

            predicted = None
            for sub_dir in sub_dirs:
                predicted_file = os.path.join(sub_dir, args.file)
                tmp = pd.read_csv(predicted_file, index_col=0)

                if predicted is None:
                    predicted = tmp
                else:
                    predicted = predicted + tmp
                
            predicted_hist = predicted / float(len(sub_dirs))

        assert len(original_hist) == len(predicted_hist)

        # Hellinger distance between two histograms
        dist = hellinger(original_hist, predicted_hist)

        # Store results
        key = os.path.basename(predicted_dir)
        results[key] = {'dist': dist, 'hist': predicted_hist}

    # Print and save results
    keys = natural_sort(results.keys())

    result_str = ['Hellinger distances:']
    for key in keys:
        result_str.append('\t{}: {}'.format(key, results[key]['dist']))
    
    result_str = '\n'.join(result_str)
    
    print(result_str)
    print('')

    # Make output directory
    out = os.path.join(args.out, os.path.basename(args.predicted))
    if not os.path.isdir(out):
        os.makedirs(out)
    
    with open(os.path.join(out, 'distances.txt'), 'w') as f:
        f.write(result_str)

    if args.visualize:
        # Set color and style
        mpl_default = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                       '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                       '#bcbd22', '#17becf']
        sns.set(context='poster', palette=sns.color_palette(mpl_default), font_scale=1.05)
        sns.set_style('white', {'legend.frameon':True})

        index = original_hist
        bins = [format(i, '.2f') for i in list(index)]

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        # Convert frequency to percentage
        gt_handle, = ax.plot(bins, normalize(original_hist) * 100, color='C4')

        # Awesome way to create a tabular-style legend
        # https://stackoverflow.com/questions/25830780/tabular-legend-layout-for-matplotlib
        # Create a blank rectangle
        blank = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)

        # Correspond to each columns of the tabular
        legend_handles = [blank, gt_handle]
        legend_names = ['Name', 'Ground Truth']
        legend_dists = ['Hell. Dist.', '0'.center(16)]

        colors = ['C1', 'C3', 'C0', 'C2'] if len(keys) <= 4 else \
                 ['C1', 'C0', 'C6', 'C7', 'C8', 'C9', 'C5', 'C2', 'C3', 'C4']
        
        assert len(keys) <= len(colors)

        for color, key in zip(colors, keys):
            predicted_hist = results[key]['hist'][:-4]
            label = key

            if 'Aud2Pose' in label:
                label += ' [14]'

            handle, = ax.plot(bins, normalize(predicted_hist) * 100, color=color)

            legend_handles.append(handle)
            legend_names.append(label)
            legend_dists.append('{:.3f}'.format(results[key]['dist']).center(12))

        # Legend will have a tabular of (rows x 3)
        rows = len(legend_handles)
        empty_label = ['']

        legend_handles = legend_handles + [blank] * (rows * 2)
        legend_labels = np.concatenate([empty_label * rows, legend_names, legend_dists])

        ax.legend(legend_handles, legend_labels,
                  ncol=3, handletextpad=0.5, columnspacing=-2.15,
                  labelspacing=0.35)
        ax.set_xlabel('Velocity (cm/s)')
        ax.set_ylabel('Frequency (%)')
        ax.set_xticks(np.arange(16))
        ax.tick_params(pad=6)
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins='auto', steps=[1, 2, 2.5, 5, 10], integer=True))

        plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.12)
        plt.savefig(os.path.join(out, 'velocity_histogram.pdf'))
        plt.show()
    
    print('Results were writen in ' + out)
    print('')


if __name__ == '__main__':
    main()
