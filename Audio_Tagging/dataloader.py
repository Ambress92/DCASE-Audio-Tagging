from scipy.io import wavfile
import tqdm
import numpy as np
import os
import tensorflow as tf

def get_verified_files_dict():
    with open('../datasets/train_curated.csv', 'r') as in_file:
        data_config = in_file.readlines()
        data_config = data_config[1:]

    verified_files_dict = {line.split(',')[0]: line.split(',')[1].replace('"', '').split(',') for line in data_config
                           if line.split(',')[2].rstrip() == '0'}

    return verified_files_dict

def get_unverified_files_dict():
    with open('../datasets/train_noisy.csv', 'r') as in_file:
        data_config = in_file.readlines()
        data_config = data_config[1:]

    unverified_files_dict = {line.split(',')[0]: line.split(',')[1].replace('"', '').split(',') for line in data_config
                             if line.split(',')[2].rstrip() == '1'}

    return unverified_files_dict

def get_test_files_list():
    """
    Determines the ground-truth label of test audio samples.

    Parameters
    ----------
    year : int
        Which year the data is to be taken from.

    Returns
    -------
    test_files : dictionary
        List containing the name of an audio sample
    """
    test_files = os.listdir('../datasets/test')

    return test_files


def load_verified_files(features=None):
    verified_files_dict = get_verified_files_dict()

    # load verified audio clips
    verified_files = []
    for file, label in tqdm.tqdm(zip(verified_files_dict.keys(), verified_files_dict.values()), 'Loading verified clips'):
        if not features:
            _, data = wavfile.read('../datasets/train_curated/{}'.format(file))
        else:
            data = np.load('../features/{}/train_curated/{}.npy'.format(features, file.replace('wav', features)))

        verified_files.append((data, label))

    return verified_files

def load_unverified_files(features=None):
    unverified_files_dict = get_unverified_files_dict()

    # load verified audio clips
    unverified_files = []
    for file, label in tqdm.tqdm(zip(unverified_files_dict.keys(), unverified_files_dict.values()),
                                 'Loading verified clips'):
        if not features:
            _, data = wavfile.read('../datasets/train_curated/{}'.format(file))
        else:
            data = np.load('../features/{}/train_curated/{}.npy'.format(features, file.replace('wav', features)))

        unverified_files.append((data, label))

    return unverified_files

def load_test_files(features=None):
    """
    Loads and returns test audio files of given year.

    Parameters
    ----------
    year : int
        Which year the data is to be taken from.
    features : String
        String containing name of feature that should be loaded.
        If 'None', raw data is loaded.

    Returns
    -------
    test_files : List of Tuples
        List containing (data, label) tupels for all test audio clips.
    """
    files = get_test_files_list()

    # load test clips
    test_files = []
    for file in tqdm.tqdm(files, 'Loading test clips'):
        if not features:
            _, data = wavfile.read('../datasets/test/{}'.format(file))
        else:
            if not os.path.exists('../features/{}/audio_test'.format(features)):
                print('\nPlease extract features prior to loading!\n')
                return
            data = np.load('../features/{}/audio_test/{}.npy'.format(features, file.replace('wav', features)))

        test_files.append(data)

    return test_files

def get_label_mapping():
    with open('../datasets/train_curated.csv', 'r') as in_file:
        train_list = in_file.readlines()

    train_list = train_list[1:]
    labels = np.unique([line.split(',')[1] if not '"' in line else line.replace('"', '').split(',') for line in train_list])


    label_mapping = {label: index for index, label in enumerate(labels)}
    inv_label_mapping = {v: k for k, v in zip(label_mapping.keys(), label_mapping.values())}
    return label_mapping, inv_label_mapping

def one_hot_encode(label, num_classes):
    """
    Derives the one-hot-encoding representation of defined
    label for given number of classes.

    Parameters
    ----------
    label : int
        Label to be one-hot-encoded.
    num_classes : int
        Total number of different classes.

    Returns
    -------
    encoding : Array
        Array containing one-hot-encoding, where label-th
        value is set to 1., and remaining values are 0.
    """

    encoding = np.zeros((num_classes))
    encoding[label] = 1
    return encoding
