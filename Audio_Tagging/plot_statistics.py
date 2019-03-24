import matplotlib.pyplot as plt
import argparse
from dataloader import get_verified_files_dict, get_unverified_files_dict
import numpy as np
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('-year', required=True)
args = parser.parse_args()

def plot_class_distribution():
    verified_files = get_verified_files_dict(args.year)
    unique_verified = np.unique(list(verified_files.values()))
    print(len(unique_verified))
    counter = Counter(verified_files.values())
    counts_verified = [counter[label] for label in unique_verified]

    unverified_files = get_unverified_files_dict(args.year)
    unique_unverified = np.unique(list(unverified_files.values()))
    print(len(unique_unverified))
    counter = Counter(unverified_files.values())
    counts_unverified = [counter[label] for label in unique_unverified]

    plt.figure(figsize=(15,15))
    plt.bar(unique_verified, counts_verified, label='verified files')
    plt.bar(unique_unverified, counts_unverified, label='unverified files')
    plt.xticks(range(len(unique_verified)), unique_verified, rotation='vertical')
    plt.title('Class distribution')
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.gcf().savefig('plots/class_distribution.png')
    plt.close()

def main():
    plot_class_distribution()

if __name__ == '__main__':
    main()