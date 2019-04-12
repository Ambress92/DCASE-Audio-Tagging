import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from dataloader import get_verified_files_dict, get_unverified_files_dict, get_total_file_dict
import numpy as np
from collections import Counter
from scipy.io import wavfile
import os

def plot_spectrogram_lengths(feature, path, idx):
    """
    Plots and saves the lengths of different features in a histogram.

    Parameters
    ----------
    feature : string
        Name of feature we take a look at, used for plotting
        and resulting file-name. If we want to plot raw data,
        this string should contain 'raw'.
    path : string
        Path to the directory containing curated and noisy
        folders of the features we want to plot lengths of.
    idx : int
        Index of length of interest within the shape of the feature.
    """
    lengths = []
    if 'raw' in feature.lower():
        for file in os.listdir(path+'/train_curated/'):
            _, data = wavfile.read(path+'/train_curated/'+file)
            lengths.append(data.shape)
        for file in os.listdir(path+'/train_noisy/'):
            _, data = wavfile.read(path+'/train_noisy/'+file)
            lengths.append(data.shape)
    else:
        for file in os.listdir(path+'/train_curated/'):
            lengths.append(np.load(path+'/train_curated/'+file).shape)
        for file in os.listdir(path+'/train_noisy/'):
            lengths.append(np.load(path+'/train_noisy/'+file).shape)
    lengths = np.asarray(lengths)
    plt.figure(figsize=(10,10))
    plt.xlabel('Lenghts')
    plt.ylabel('Occurences')
    plt.title('Distribution of {} lengths'.format(feature))
    n, bins, patches = plt.hist(lengths[:, idx], color=cm.get_cmap('viridis')(0.))
    plt.axvline(x=np.median(lengths[:, idx]), color=cm.get_cmap('viridis')(1.), linestyle='dashed', linewidth=2)
    print('Maximum: ' + str(max(lengths[:, idx])))
    print('Minimum: ' + str(min(lengths[:, idx])))
    print('Median: ' + str(np.median(lengths[:, idx])))
    plt.gcf().savefig('plots/{}_lengths.png'.format(feature))
    plt.close()


def save_stacked_bar_plot(counts_verified, labels_verified, counts_unverified, start, end, exp_name):
    plt.figure(figsize=(10,10))
    plt.bar(np.arange(len(counts_verified)), counts_verified, label='verified files')
    plt.bar(np.arange(len(counts_unverified)), counts_unverified, label='unverified files', bottom=counts_verified)
    plt.xticks(range(len(labels_verified)), labels_verified, rotation='vertical')
    plt.title('Class distribution')
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.gcf().savefig('plots/{}_{}_{}.png'.format(exp_name, start, end))
    plt.close()


def plot_class_distribution():
    verified_files = get_verified_files_dict()
    labels = []
    for label in verified_files.values():
        current_labels = [l for l in label]
        labels.extend(current_labels)
    unique_verified = np.unique(labels)
    print('Number of curated labels: ', len(unique_verified))
    counter = Counter(labels)
    counts_verified = [counter[label] for label in unique_verified]

    unverified_files = get_unverified_files_dict()
    labels = []
    for label in unverified_files.values():
        current_labels = [l for l in label]
        labels.extend(current_labels)
    unique_unverified = np.unique(labels)
    print('Number of noisy labels: ', len(unique_unverified))
    counter = Counter(labels)
    counts_unverified = [counter[label] for label in unique_unverified]

    interval = 20
    start = 0
    while start != len(unique_verified):
        save_stacked_bar_plot(counts_verified[start:start+interval], unique_verified[start:start+interval],
                              counts_unverified[start:start+interval], start, start+interval, 'class_distribution')
        start += interval


def plot_fold_distribution(fold):
    file_dict = get_total_file_dict()
    label_dict = {}

    with open('../datasets/cv/fold{}'.format(fold), 'r') as in_file:
        filelist = in_file.readlines()

    for file in filelist:
        file = file.rstrip()
        labels = file_dict[file+'.wav']
        if len(labels) > 1:
            label_counts = {}
            for l in labels:
                if not l in label_dict.keys():
                    label_counts[l] = 0
                else:
                    label_counts[l] = label_dict[l]
            # get labels with minimum classes present and append file with multilabel to this class
            min_element_label = min(label_counts.items(), key=lambda x: x[1])[0]

            if not min_element_label in label_dict.keys():
                label_dict[min_element_label] = 1
            else:
                label_dict[min_element_label] += 1
        else:
            label = labels[0]
            if not label in label_dict.keys():
                label_dict[label] = 1
            else:
                label_dict[label] += 1

    labels = label_dict.keys()
    plt.figure(figsize=(10, 10))
    plt.bar(np.arange(len(label_dict)), label_dict.values())
    plt.xticks(range(len(labels)), labels, rotation='vertical')
    plt.title('Class distribution of fold {}'.format(fold))
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.gcf().savefig('plots/class_distribution_fold_{}.png'.format(fold))
    plt.close()

def plot_single_label_dist():
    verified_files = get_verified_files_dict()
    unverified_files = get_unverified_files_dict()

    verified_labels = []
    for labels in verified_files.values():
        if len(labels) > 1:
            continue
        verified_labels.append(labels[0])

    unverified_labels = []
    for labels in unverified_files.values():
        if len(labels) > 1:
            continue
        unverified_labels.append(labels[0])

    label_list = np.unique(unverified_labels)
    counter_verified = Counter(verified_labels)
    counts_verified = [counter_verified[l] for l in label_list]
    counter_noisy = Counter(unverified_labels)
    counts_noisy = [counter_noisy[l] for l in label_list]

    interval = 20
    start = 0
    while start != len(label_list):
        save_stacked_bar_plot(counts_verified[start:start + interval], label_list[start:start + interval],
                              counts_noisy[start:start + interval], start, start + interval, 'single_class_distribution')
        start += interval

def plot_multi_label_dist():
    verified_files = get_verified_files_dict()
    unverified_files = get_unverified_files_dict()

    verified_labels = []
    for labels in verified_files.values():
        if len(labels) == 1:
            continue
        verified_labels.extend([l for l in labels])

    unverified_labels = []
    for labels in unverified_files.values():
        if len(labels) == 1:
            continue
        unverified_labels.extend([l for l in labels])

    label_list = np.unique(unverified_labels)
    counter_verified = Counter(verified_labels)
    counts_verified = [counter_verified[l] for l in label_list]
    counter_noisy = Counter(unverified_labels)
    counts_noisy = [counter_noisy[l] for l in label_list]

    interval = 20
    start = 0
    while start != len(label_list):
        save_stacked_bar_plot(counts_verified[start:start + interval], label_list[start:start + interval],
                              counts_noisy[start:start + interval], start, start + interval, 'multi_class_distribution')
        start += interval

def main():
    plot_class_distribution()
    for fold in range(1,5):
        plot_fold_distribution(fold)
    plot_single_label_dist()
    plot_multi_label_dist()

if __name__ == '__main__':
    main()
