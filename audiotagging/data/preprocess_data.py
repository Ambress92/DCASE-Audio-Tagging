import os
import numpy as np

from sequence import read_label_dict

np.random.seed(101)


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def split_train_val():
    """
    Creates train/validation splits for
    :return:
    """
    splits = [[1, 2, 3], [2, 3, 4], [1, 2, 4], [1, 3, 4]]
    eval = [4, 1, 3, 2]
    for set in ['curated', 'noisy']:
        for i, split in enumerate(splits):
            with open('../../datasets/cv/fold{}_{}_train'.format(i + 1, set), 'w') as train_out:
                for fold in split:
                    with open('../../datasets/cv/fold{}_{}'.format(fold, set), 'r') as in_file:
                        files = in_file.read()
                    train_out.write(files)
            with open('../../datasets/cv/fold{}_{}'.format(eval[i], set), 'r') as val_in:
                val_files = val_in.read()
            with open('../../datasets/cv/fold{}_{}_eval'.format(i+1, set), 'w') as val_out:
                val_out.write(val_files)


def add_noisy_training_data():
    """
    Creates extended train splits including noisy files.
    :return:
    """
    for i in range(1, 5):
        with open('../../datasets/cv/fold{}_train'.format(i), 'r') as in_file:
            curated = in_file.read()
        noisy = read_label_dict('../../datasets/train_noisy.csv').keys()
        noisy = ''.join(noisy).replace('.wav', '\n')

        with open('../../datasets/cv/fold{}_train_both'.format(i), 'w') as out_file:
            out_file.write(curated+noisy)


def create_stratified_cv_splits():
    """
    Creates a stratified cross-validation setup and stores a list of files for each fold
    :return:
    """
    curated_file_dict = read_label_dict('../../datasets/train_curated.csv')
    noisy_file_dict = read_label_dict('../../datasets/train_noisy.csv')

    print('Number of curated labels: ', len(curated_file_dict))
    print('Number of noisy labels: ', len(noisy_file_dict))

    curated_label_dict = {}
    for file, labels in curated_file_dict.items():
        # discard files that were empty after silence clipping and did not yield any spectrograms
        if os.path.exists('../../features/mel/train_curated/{}'.format(file.replace('.wav', '.npy'))):
            if len(labels) > 1:
                label_counts = {}
                for l in labels:
                    if not l in curated_label_dict.keys():
                        label_counts[l] = 0
                    else:
                        label_counts[l] = len(curated_label_dict[l])
                # get labels with minimum classes present and append file with multilabel to this class
                min_element_label = min(label_counts.items(), key=lambda x: x[1])[0]
                if not min_element_label in curated_label_dict.keys():
                    curated_label_dict[min_element_label] = [file]
                else:
                    curated_label_dict[min_element_label].append(file)
            else:
                label = labels[0]
                if not label in curated_label_dict.keys():
                    curated_label_dict[label] = [file]
                else:
                    curated_label_dict[label].append(file)

    noisy_label_dict = {}
    for file, labels in noisy_file_dict.items():
        # discard files that were empty after silence clipping and did not yield any spectrograms
        if os.path.exists('../../features/mel/train_noisy/{}'.format(file.replace('.wav', '.npy'))):
            if len(labels) > 1:
                label_counts = {}
                for l in labels:
                    if not l in noisy_label_dict.keys():
                        label_counts[l] = 0
                    else:
                        label_counts[l] = len(noisy_label_dict[l])
                # get labels with minimum classes present and append file with multilabel to this class
                min_element_label = min(label_counts.items(), key=lambda x: x[1])[0]
                if not min_element_label in noisy_label_dict.keys():
                    noisy_label_dict[min_element_label] = [file]
                else:
                    noisy_label_dict[min_element_label].append(file)
            else:
                label = labels[0]
                if not label in noisy_label_dict.keys():
                    noisy_label_dict[label] = [file]
                else:
                    noisy_label_dict[label].append(file)

    labels = list(curated_label_dict.keys())
    # randomly shuffle curated and noisy labels
    # overall_dict = {}
    for l in labels:
        #overall_dict[l] = curated_label_dict[l]
        #overall_dict[l].extend(noisy_label_dict[l])
        np.random.shuffle(curated_label_dict[l])
        np.random.shuffle(noisy_label_dict[l])
        # print("{} Samples for class {} in dictionary".format(len(overall_dict[l]), l))

    if not os.path.exists('../../datasets/cv'):
        os.makedirs('../../datasets/cv')

    for l in labels:
        length_curated = int(len(curated_label_dict[l])/4)+1
        chunks_curated = list(divide_chunks(curated_label_dict[l], length_curated))
        length_noisy = int(len(noisy_label_dict[l]) / 4) + 1
        chunks_noisy = list(divide_chunks(noisy_label_dict[l], length_noisy))
        for f, chunk in enumerate(range(4)):
            for file in chunks_curated[chunk]:
                with open('../../datasets/cv/fold{}_curated'.format(f+1), 'a+') as out_file:
                    out_file.write(file.replace('.wav', '') + '\n')

            for file in chunks_noisy[chunk]:
                with open('../../datasets/cv/fold{}_noisy'.format(f+1), 'a+') as out_file:
                        out_file.write(file.replace('.wav', '') + '\n')

    split_train_val()


def create_stratified_curated_cv_splits():
    """
    Creates a stratified cross-validation setup and stores a list of files for each fold
    :return:
    """
    curated_file_dict = read_label_dict('../../datasets/train_curated.csv')
    print('Number of curated labels: ', len(curated_file_dict))

    curated_label_dict = {}
    for file, labels in curated_file_dict.items():
        # discard files that were empty after silence clipping and did not yield any spectrograms
        if os.path.exists('../../features/mel/train_curated/{}'.format(file.replace('.wav', '.npy'))):
            if len(labels) > 1:
                label_counts = {}
                for l in labels:
                    if not l in curated_label_dict.keys():
                        label_counts[l] = 0
                    else:
                        label_counts[l] = len(curated_label_dict[l])
                # get labels with minimum classes present and append file with multilabel to this class
                min_element_label = min(label_counts.items(), key=lambda x: x[1])[0]
                if not min_element_label in curated_label_dict.keys():
                    curated_label_dict[min_element_label] = [file]
                else:
                    curated_label_dict[min_element_label].append(file)
            else:
                label = labels[0]
                if not label in curated_label_dict.keys():
                    curated_label_dict[label] = [file]
                else:
                    curated_label_dict[label].append(file)

    labels = list(curated_label_dict.keys())
    # randomly shuffle curated labels
    for l in labels:
        np.random.shuffle(curated_label_dict[l])

    if not os.path.exists('../../datasets/cv'):
        os.makedirs('../../datasets/cv')

    for l in labels:
        length_curated = int(len(curated_label_dict[l])/4)+1
        chunks_curated = list(divide_chunks(curated_label_dict[l], length_curated))
        for f, chunk in enumerate(range(4)):
            for file in chunks_curated[chunk]:
                with open('../../datasets/cv/fold{}'.format(f+1), 'a+') as out_file:
                    out_file.write(file.replace('.wav', '') + '\n')

    splits = [[1, 2, 3], [2, 3, 4], [1, 2, 4], [1, 3, 4]]
    eval = [4, 1, 3, 2]
    for i, split in enumerate(splits):
        with open('../../datasets/cv/fold{}_train'.format(i + 1), 'w') as train_out:
            for fold in split:
                with open('../../datasets/cv/fold{}'.format(fold), 'r') as in_file:
                    files = in_file.read()
                train_out.write(files)
        with open('../../datasets/cv/fold{}'.format(eval[i]), 'r') as val_in:
            val_files = val_in.read()
        with open('../../datasets/cv/fold{}_eval'.format(i+1), 'w') as val_out:
            val_out.write(val_files)
    add_noisy_training_data()


if __name__ == '__main__':
    create_stratified_curated_cv_splits()
