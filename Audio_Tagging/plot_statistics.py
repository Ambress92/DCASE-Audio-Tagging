import matplotlib.pyplot as plt
import argparse
from dataloader import get_verified_files_dict, get_unverified_files_dict, get_total_file_dict
import numpy as np
from collections import Counter

def save_stacked_bar_plot(counts_verified, labels_verified, counts_unverified, start, end):
    plt.figure(figsize=(10,10))
    plt.bar(np.arange(len(counts_verified)), counts_verified, label='verified files')
    plt.bar(np.arange(len(counts_unverified)), counts_unverified, label='unverified files', bottom=counts_verified)
    plt.xticks(range(len(labels_verified)), labels_verified, rotation='vertical')
    plt.title('Class distribution')
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.gcf().savefig('plots/class_distribution_{}_{}.png'.format(start, end))
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
                              counts_unverified[start:start+interval], start, start+interval)
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




def main():
    # plot_class_distribution()
    plot_fold_distribution(4)

if __name__ == '__main__':
    main()