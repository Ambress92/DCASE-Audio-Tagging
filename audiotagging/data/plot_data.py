import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from scipy.io import wavfile
from collections import Counter
from sequence import read_label_dict


def plot_spectrogram_lengths(fold=None, feature='mel', idx=1):
    """
    Plots and saves the lengths of different features in a histogram.

    Parameters
    ----------
    fold : string
        Name of fold we want to take a look at.
        If `None`, looks at all files. Defaults to `None`.
    feature : string
        Name of feature we take a look at, used for plotting
        and resulting file-name. If we want to plot raw data,
        this string should contain 'raw'. Defaults to `mel`.
    idx : int
        Index of length of interest within the shape of the feature.
        1 for Mel-related features, 0 otherwise. Defaults to `1`.
    """
    if fold is None:
        files = [file.rsplit('.wav')[0]
                 for file in os.listdir('../../datasets/train_curated/') if 'clipped' not in file]
        files.extend([file.rsplit('.wav')[0]
                      for file in os.listdir('../../datasets/train_noisy/') if 'clipped' not in file])
    else:
        with open('../../datasets/cv/fold{}'.format(fold), 'r') as fp:
            files = fp.readlines()
            files = np.array([file.rstrip() for file in files])

    lengths = []
    for file in files:
        if 'raw' in feature.lower():
            data = np.empty(0)
            try:
                _, data = wavfile.read(os.path.join('../../datasets/train_curated/', file+'.wav'))
            except FileNotFoundError:
                _, data = wavfile.read(os.path.join('../../datasets/train_noisy/', file+'.wav'))
            finally:
                lengths.append(data.shape)
        else:
            data = np.empty(0)
            try:
                data = np.load(os.path.join('../../features/'+feature+'/train_curated/', file+'.npy'))
            except FileNotFoundError:
                data = np.load(os.path.join('../../features/'+feature+'/train_noisy/', file+'.npy'))
            finally:
                lengths.append(data.shape)
    lengths = np.asarray(lengths)

    plt.figure(figsize=(10,10))
    plt.xlabel('Lengths')
    plt.ylabel('Occurences')
    plt.title('Distribution of {} lengths'.format(feature))
    n, bins, patches = plt.hist(lengths[:, idx], color=cm.get_cmap('viridis')(0.))
    plt.axvline(x=np.median(lengths[:, idx]), color=cm.get_cmap('viridis')(1.), linestyle='dashed', linewidth=2)
    print('Maximum: ' + str(max(lengths[:, idx])))
    print('Minimum: ' + str(min(lengths[:, idx])))
    print('Median: ' + str(np.median(lengths[:, idx])))
    if fold is not None:
        plt.gcf().savefig('../../plots/{}_fold{}_lengths.png'.format(feature, fold))
    else:
        plt.gcf().savefig('../../plots/{}_lengths.png'.format(feature))
    plt.close()


def plot_func_per_fold(func, **kwargs):
    for fold in range(1, 5):
        func(fold, **kwargs)


def plot_fold_distribution(fold):
    file_dict = read_label_dict('../../datasets/train_curated.csv')
    file_dict.update(read_label_dict('../../datasets/train_noisy.csv'))
    labels_s = sorted({l for ll in file_dict.values() for l in ll})
    label_dict = {l: 0 for l in labels_s}

    with open('../../datasets/cv/fold{}'.format(fold), 'r') as in_file:
        filelist = in_file.readlines()

    for file in filelist:
        file = file.rstrip()
        labels = file_dict[file+'.wav']
        if len(labels) > 1:
            label_counts = {}
            for l in labels:
                label_counts[l] = label_dict[l]
            # get labels with minimum classes present and append file with multilabel to this class
            min_element_label = min(label_counts.items(), key=lambda x: x[1])[0]

            if not min_element_label in label_dict.keys():
                label_dict[min_element_label] = 1
            else:
                label_dict[min_element_label] += 1
        else:
            label = labels[0]
            label_dict[label] += 1

    labels = label_dict.keys()
    plt.figure(figsize=(10, 10))
    plt.bar(np.arange(len(label_dict)), label_dict.values())
    plt.xticks(range(len(labels)), labels, rotation='vertical')
    plt.title('Class distribution of fold {}'.format(fold))
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.gcf().savefig('../../plots/class_distribution_fold_{}.png'.format(fold))
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
    plt.gcf().savefig('../../plots/{}_{}_{}.png'.format(exp_name, start, end))
    plt.close()


def plot_single_label_dist():
    verified_files = read_label_dict('../../datasets/train_curated.csv')
    unverified_files = read_label_dict('../../datasets/train_noisy.csv')

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
    verified_files = read_label_dict('../../datasets/train_curated.csv')
    unverified_files = read_label_dict('../../datasets/train_noisy.csv')

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


def plot_class_distribution():
    verified_files = read_label_dict('../../datasets/train_curated.csv')
    labels = []
    for label in verified_files.values():
        current_labels = [l for l in label]
        labels.extend(current_labels)
    unique_verified = np.unique(labels)
    print('Number of curated labels: ', len(unique_verified))
    counter = Counter(labels)
    counts_verified = [counter[label] for label in unique_verified]

    unverified_files = read_label_dict('../../datasets/train_noisy.csv')
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


if __name__ == '__main__':
    if not os.path.isdir('../../plots/'):
        os.mkdir('../../plots/')

    plot_func_per_fold(plot_fold_distribution)