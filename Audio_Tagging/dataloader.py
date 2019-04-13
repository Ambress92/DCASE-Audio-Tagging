from scipy.io import wavfile
import tqdm
import numpy as np
import os

def repeat_spectrogram(spec, fixed_length):
    if spec.shape[1] < fixed_length:
        while spec.shape[1] < fixed_length:
            spec = np.concatenate((spec, spec), axis=-1)

    if spec.shape[1] > fixed_length:
        spec = spec[:, :fixed_length]

    return spec


def divide_chunks(l, n):
    for i in range(0, l.shape[1], n):
        yield l[:, i:i + n]

def sample_from_spec(spec, frame_size, n_frames):
    # sample frames of spectrogram randomly across the whole spectrogram
    frame_range = np.arange(0,spec.shape[1]-frame_size)
    start_idxs = np.random.choice(frame_range, n_frames)
    for idx in start_idxs:
        yield spec[:, idx:idx+frame_size]


def get_verified_files_dict():
    with open('../datasets/train_curated.csv', 'r') as in_file:
        data_config = in_file.readlines()
        data_config = data_config[1:]

    verified_files_dict = {line.split(',')[0]: line[line.index(',')+1:].rstrip().replace('"', '').split(',')
                           for line in data_config}

    return verified_files_dict

def get_unverified_files_dict():
    with open('../datasets/train_noisy.csv', 'r') as in_file:
        data_config = in_file.readlines()
        data_config = data_config[1:]

    unverified_files_dict = {line.split(',')[0]: line[line.index(',')+1:].rstrip().replace('"', '').split(',')
                             for line in data_config}

    return unverified_files_dict

def get_total_file_dict():
    curated_files = get_verified_files_dict()
    noisy_files = get_unverified_files_dict()
    return dict(curated_files, **noisy_files)

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


def load_test_features(filelist, features, fixed_length=3480, n_frames=10):
    """
        Loads and returns test audio files.

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
    X = []
    for file in filelist:
        data = np.load(
            '../features/{}/test/{}.npy'.format(features, file.rstrip().replace('.wav', '')))

        if features != 'mfcc':
            if data.shape[1] < fixed_length:
                # repeat spectrogram and split into frames
                data = repeat_spectrogram(data, fixed_length=fixed_length)
                data = list(divide_chunks(data, int(fixed_length / n_frames)))
            else:
                # spectrogram is too long - sample frames from spectrogram
                frame_size = int(fixed_length / n_frames)
                data = list(sample_from_spec(data, frame_size, n_frames))

        X.extend(np.asarray(data))

    return np.asarray(X)

    
def load_features(filelist, features, num_classes, fixed_length=3132, n_frames=9):
    # load verified audio clips
    curated_files_dict = get_verified_files_dict()
    noisy_files_dict = get_unverified_files_dict()
    label_mapping, inv_label_mapping = get_label_mapping()
    X = []
    y = []

    for file in filelist:
        file = file.rstrip()+'.wav'
        if file in curated_files_dict.keys():
            data = np.load(
                '../features/{}/train_curated/{}.npy'.format(features, file.rstrip().replace('.wav', '')))

            labels = curated_files_dict[file]
        else:
            data = np.load(
                '../features/{}/train_noisy/{}.npy'.format(features, file.rstrip().replace('.wav', '')))

            labels = noisy_files_dict[file]
        
        if features != 'mfcc':
            if data.shape[1] < fixed_length:
                # repeat spectrogram and split into frames
                data = repeat_spectrogram(data, fixed_length=fixed_length)
                data = list(divide_chunks(data, int(fixed_length/n_frames)))
            else:
                #spectrogram is too long - sample frames from spectrogram
                frame_size = int(fixed_length/n_frames)
                data = list(sample_from_spec(data, frame_size, n_frames))

        if len(labels) > 1:
            label = [label_mapping[l] for l in labels]
        else:
            label = label_mapping[labels[0]]

        label = one_hot_encode(np.asarray(label), num_classes)
        for i in range(len(data)):
            y.append(label)
        X.extend(np.asarray(data))

    return np.asarray(X), np.asarray(y)

def load_batches(filelist, batchsize, shuffle=False, drop_remainder=False, infinite=False):
    num_datapoints = len(filelist)
    if shuffle:
        np.random.shuffle(filelist)

    rest = (num_datapoints % batchsize)
    upper_bound = num_datapoints - (rest if drop_remainder else 0)
    for start_idx in range(0, upper_bound, batchsize):
        batch = filelist[start_idx: start_idx+batchsize]
        yield batch

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

def get_label_mapping():
    with open('../datasets/train_curated.csv', 'r') as in_file:
        train_list = in_file.readlines()

    train_list = train_list[1:]
    single_labels = []
    labels = [line[line.index(',')+1:].rstrip().replace('"', '').split(',')
                           for line in train_list]
    for label in labels:
        if len(label) > 1:
            single_labels.extend([l for l in label])
        else:
            single_labels.append(label[0])

    unique_labels = np.unique(single_labels)
    label_mapping = {label: index for index, label in enumerate(unique_labels)}
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
