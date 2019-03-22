import matplotlib.pyplot as plt
import argparse
from dataloader import get_verified_files_dict, get_unverified_files_dict
import numpy as np
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('-year', required=True)
args = parser.parse_args()

def plot_class_distribution(type):
    if type == 'verified':
        datapoints = get_verified_files_dict(args.year)
    else:
        datapoints = get_unverified_files_dict(args.year)

    unique = np.unique(list(datapoints.values()))
    print(len(unique))
    counter = Counter(datapoints.values())
    counts = [counter[label] for label in unique]
    plt.figure(figsize=(15,15))
    plt.bar(unique, counts)
    plt.xticks(range(len(unique)), unique, rotation='vertical')
    plt.title('Class distribution of {} labels'.format(type))
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.gcf().savefig('plots/class_distribution_{}.png'.format(type))
    plt.close()

def main():
    plot_class_distribution('verified')
    plot_class_distribution('unverified')

if __name__ == '__main__':
    main()