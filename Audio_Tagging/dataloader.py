from scipy.io import wavfile
import tqdm
import numpy as np
np.random.seed(101)
import os
import librosa
import feature_extractor

def sample_from_spec(spec, frame_size, feature_width):
    # sample frames of spectrogram randomly across the whole spectrogram
    frame_range = np.arange(0,spec.shape[1]-frame_size)
    start_idxs = np.random.choice(frame_range, feature_width)
    for idx in start_idxs:
        yield spec[:, idx:idx+frame_size]

def get_verified_files_dict(path='../datasets/'):
    with open(path+'train_curated.csv', 'r') as in_file:
        data_config = in_file.readlines()
        data_config = data_config[1:]

    verified_files_dict = {line.split(',')[0]: line[line.index(',')+1:].rstrip().replace('"', '').split(',')
                           for line in data_config}

    return verified_files_dict

def get_unverified_files_dict(path='../datasets/'):
    with open(path+'train_noisy.csv', 'r') as in_file:
        data_config = in_file.readlines()
        data_config = data_config[1:]

    unverified_files_dict = {line.split(',')[0]: line[line.index(',')+1:].rstrip().replace('"', '').split(',')
                             for line in data_config}

    return unverified_files_dict

def get_total_file_dict(path='../datasets/'):
    curated_files = get_verified_files_dict(path)
    noisy_files = get_unverified_files_dict(path)
    return dict(curated_files, **noisy_files)

def get_test_files_list():
    """
    Determines the ground-truth label of test audio samples.

    Returns
    -------
    test_files : dictionary
        List containing the name of an audio sample
    """
    test_files = os.listdir('../datasets/test')

    return test_files


def load_test_features(audio_clips, filelist, features, fixed_length=2784, feature_width=348, sr=32000, mixup=False):
    """
    Loads and returns test audio files.

    Parameters
    ----------
    filelist : List
        List containing names of relevant test files as strings.
    features : String
        String containing name of feature that should be loaded.
    path : String
        Path pointing to respective feature-folder.
    fixed_length : int
        Integer that restricts the final length of all features.
        Defaults to `3132`.
    feature_width : int
        Number of frames within a feature. Defaults to `9`.

    Returns
    -------
    test_files : List of Tuples
        List containing (data, label) tupels for all test audio clips.
    """
    X = []

    if features == 'mel':
        spectrograms = feature_extractor.get_mel_specs(audio_clips, filelist, sr=sr, spec_weighting=False, plot=False, dump=False,
                                                       mixup=mixup, fixed_length=fixed_length, test=True)
    elif features == 'mel_weighted':
        spectrograms = feature_extractor.get_mel_specs(audio_clips, filelist, sr=sr, spec_weighting=True,
                                                               plot=False, dump=False,
                                                               mixup=mixup, fixed_length=fixed_length, test=True)
    elif features == 'cqt':
        spectrograms = feature_extractor.get_cqt_specs(audio_clips, filelist, sr=sr, spec_weighting=False,
                                                               plot=False, dump=False,
                                                               mixup=mixup, fixed_length=fixed_length, test=True)
    elif features == 'cqt_weighted':
        spectrograms = feature_extractor.get_cqt_specs(audio_clips, filelist, sr=sr, spec_weighting=True,
                                                               plot=False, dump=False,
                                                               mixup=mixup, fixed_length=fixed_length, test=True)

    for data in spectrograms:
        data = list(divide_chunks(data, int(fixed_length / feature_width)))
        X.extend(np.asarray(data))

    return np.asarray(X)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def divide_chunks(spec, frame_length, jump):
    # Divide whole spectrogram into windows which overlap according to the jump parameter
    for j in range(0, spec.shape[1]-jump, jump):
        yield spec[:, j:j + frame_length]


def load_features(audio_clips, filelist, features, num_classes,
                    data_path='../datasets/', fixed_length=2784, feature_width=348, sr=32000, mixup=True):
    """
    Loads and returns audio features and their respective labels.

    Parameters
    ----------
    filelist : List
        List containing names of relevant files as strings.
    features : String
        String containing name of feature that should be loaded.
    num_classes : int
        Number of possible labels.
    feature_path : String
        Path pointing to respective feature-folder.
    data_path : String
        Path pointing to `train_curated.csv` and `train_noisy.csv`.
    fixed_length : int
        Integer that restricts the final length of all features.
        Defaults to `3132`.
    feature_width : int
        Number of frames within a feature. Defaults to `9`.

    Returns
    -------
    X : Array
        Array containing loaded features for all files in filelist.
    y : Array
        Array containing labels for all files in filelist.
    """
    # load verified audio clips
    label_mapping, inv_label_mapping = get_label_mapping(data_path)
    X = []
    y = []

    labels = audio_clips[1]
    labels_encoded = []
    for label in labels:
        l = [label_mapping[l] for l in label]
        one_hot = one_hot_encode(np.asarray(l), num_classes)
        labels_encoded.append(one_hot)
    labels_encoded = np.asarray(labels_encoded)
    audio_clips = (audio_clips[0], labels_encoded)

    if features == 'mel':
        spectrograms, labels = feature_extractor.get_mel_specs(audio_clips, filelist, sr=sr, spec_weighting=False, plot=False, dump=False,
                                                       mixup=mixup, fixed_length=fixed_length)
    elif features == 'mel_weighted':
        spectrograms, labels = feature_extractor.get_mel_specs(audio_clips, filelist, sr=sr, spec_weighting=True,
                                                               plot=False, dump=False,
                                                               mixup=mixup, fixed_length=fixed_length)
    elif features == 'cqt':
        spectrograms, labels = feature_extractor.get_cqt_specs(audio_clips, filelist, sr=sr, spec_weighting=False,
                                                               plot=False, dump=False,
                                                               mixup=mixup, fixed_length=fixed_length)
    elif features == 'cqt_weighted':
        spectrograms, labels = feature_extractor.get_cqt_specs(audio_clips, filelist, sr=sr, spec_weighting=True,
                                                               plot=False, dump=False,
                                                               mixup=mixup, fixed_length=fixed_length)

    for data, label in zip(spectrograms, labels):
        data = list(divide_chunks(data, feature_width, feature_width//2))
        for i in range(len(data)):
            y.append(label)
        X.extend(np.asarray(data))

    return np.asarray(X), np.asarray(y)

def event_oversampling(X, feature_width=348):
    batch_size, h, w, c = X.shape
    X_new = np.zeros((batch_size, h, feature_width, c), dtype=np.float32)
    for i in range(batch_size):
        # compute frame sample probabilities
        sample_probs = X[i, :, :, :].mean(axis=(0,2))
        sample_probs -= sample_probs.min()
        sample_probs /= sample_probs.sum()

        # sample center frame
        center_frame = np.random.choice(range(X.shape[2]), p = sample_probs)

        # set sample window
        start = center_frame - feature_width // 2
        start = np.clip(start, 0, X.shape[2] - feature_width)
        stop = start + feature_width

        X_new[i] = X[i, :, :, start:stop]

        return X_new

def load_batches(filelist, batchsize, feature_path='../features/', data_path='../datasets/',
                 shuffle=False, drop_remainder=False, infinite=False, num_classes=80, features='mel', test=False,
                 augment=True, feature_width=348, fixed_length=2784, sr=32000, mixup=True):
    num_datapoints = len(filelist)
    curated_files = get_verified_files_dict(data_path)

    while True:

        if shuffle:
            np.random.shuffle(filelist)

        rest = (num_datapoints % batchsize)
        upper_bound = num_datapoints - (rest if drop_remainder else 0)
        for start_idx in range(0, upper_bound, batchsize):
            batch = filelist[start_idx: start_idx+batchsize]

            if not test:
                if batch[0].rstrip()+'.wav' in curated_files.keys():
                    audio_clips = load_verified_files(batch, sr=sr, silence_clipping=True)
                else:
                    audio_clips = load_unverified_files(batch, sr=sr, silence_clipping=True)

                X, y = load_features(audio_clips, batch, features=features, num_classes=num_classes,
                                     data_path=data_path, fixed_length=fixed_length,
                                     feature_width=feature_width, sr=sr, mixup=mixup)
                X = X[:,:,:,np.newaxis]

                yield (X, y)
            else:
                X = load_test_features(batch, features, path=feature_path)
                X = X[:,:,:,np.newaxis]
                yield X

        if not infinite:
            break

def load_batches_verification(filelist, feature_path='../features/', data_path='../datasets/',
                 shuffle=False, drop_remainder=False, infinite=False, num_classes=80, features='mel', k=24, feature_width=348,
                    fixed_length=2784, sr=32000, mixup=False):
    num_datapoints = len(filelist)
    curated_files = get_verified_files_dict(data_path)

    while True:

        if shuffle:
            np.random.shuffle(filelist)

        rest = (num_datapoints % k)
        upper_bound = num_datapoints - (rest if drop_remainder else 0)
        for start_idx in range(0, upper_bound, k):
            X_train = []
            y_train = []
            for file in filelist[start_idx:start_idx+k]:

                if file in curated_files.keys():
                    audio_clips = load_verified_files([file], sr=sr, silence_clipping=True)
                else:
                    audio_clips = load_unverified_files([file], sr=sr, silence_clipping=True)

                X_temp, y_temp = load_features([audio_clips], filelist, features=features, num_classes=num_classes,
                                     data_path=data_path, fixed_length=fixed_length,
                                     feature_width=feature_width, sr=sr, mixup=mixup)

                rand_ind = np.random.choice(X_temp.shape[0])
                X_train.append(X_temp[rand_ind])
                y_train.append(y_temp[rand_ind])

            yield (np.asarray(X_train)[:,:,:,np.newaxis], np.asarray(y_train))

        if not infinite:
            break

def load_verified_files(filelist, sr, features=None, silence_clipping=True):
    verified_files_dict = get_verified_files_dict()

    # load verified audio clips
    datapoints = []
    labels = []
    for file in filelist:
        file = file.rstrip()+'.wav'
        label = verified_files_dict[file]
        if not features:
            if silence_clipping:
                # perform silence clipping
                aug_cmd = "norm -0.1 silence 1 0.025 0.15% norm -0.1 reverse silence 1 0.025 0.15% reverse"
                aug_audio_file = file.replace('.wav', '_clipped.wav')
                command = "sox %s %s %s" % (
                '../datasets/train_curated/{}'.format(file), '../datasets/train_curated/{}'.format(aug_audio_file),
                aug_cmd)
                os.system(command)

                assert os.path.exists(
                    '../datasets/train_curated/{}'.format(aug_audio_file)), "SOX Problem ... clipped wav does not exist!"

                data, sr = librosa.load('../datasets/train_curated/{}'.format(aug_audio_file), sr=sr, mono=True)

                if len(data) == 0:
                    data, sr = librosa.load('../datasets/train_curated/{}'.format(file), sr=sr, mono=True)
            else:
                data, sr = librosa.load('../datasets/train_curated/{}'.format(file), sr=sr)

            os.remove('../datasets/train_curated/{}'.format(aug_audio_file))
        else:
            data = np.load('../features/{}/{}.npy'.format(features, file.replace('wav', features)))

        datapoints.append(data)
        labels.append(label)

    return (datapoints, labels)

def load_unverified_files(filelist, sr, features=None, silence_clipping=True):
    unverified_files_dict = get_unverified_files_dict()

    # load unverified audio clips
    datapoints = []
    labels = []
    for file in filelist:
        label = unverified_files_dict[file]
        if not features:
            if silence_clipping:
                # perform silence clipping
                aug_cmd = "norm -0.1 silence 1 0.025 0.15% norm -0.1 reverse silence 1 0.025 0.15% reverse"
                aug_audio_file = file.replace('.wav', '_clipped.wav')
                command = "sox %s %s %s" % (
                    '../datasets/train_noisy/{}'.format(file),
                    '../datasets/train_noisy/{}'.format(aug_audio_file),
                    aug_cmd)
                os.system(command)

                assert os.path.exists(
                    '../datasets/train_noisy/{}'.format(
                        aug_audio_file)), "SOX Problem ... clipped wav does not exist!"

                data, sr = librosa.load('../datasets/train_noisy/{}'.format(aug_audio_file), sr=sr, mono=True)

                if len(data) == 0:
                    data, sr = librosa.load('../datasets/train_noisy/{}'.format(file), sr=sr, mono=True)

                os.remove('../datasets/train_noisy/{}'.format(aug_audio_file))
            else:
                data, sr = librosa.load('../datasets/train_noisy/{}'.format(file), sr=sr)
        else:
            data = np.load('../features/{}/{}.npy'.format(features, file.replace('wav', features)))

        datapoints.append(data)
        labels.append(label)

    return (datapoints, labels)

def get_label_mapping(path='../datasets/'):
    with open(path+'train_curated.csv', 'r') as in_file:
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

def one_hot_encode(labels, num_classes):
    """
    Derives the one-hot-encoding representation of defined
    label for given number of classes.

    Parameters
    ----------
    labels : Array
        Array of indices to be one-hot-encoded.
    num_classes : int
        Total number of different classes.

    Returns
    -------
    encoding : Array
        Array containing one-hot-encoding, where all labels-th
        values are set to 1., and remaining values are 0.
    """
    encoding = np.zeros(shape=num_classes)
    encoding[labels] = 1
    return encoding
