import os

import numpy as np
from keras.utils import Sequence
from scipy.io import wavfile


def _read_label_dict(file_path):
    data = {}
    with open(file_path) as fp:
        fp.readline()
        for line in fp:
            row = line.rstrip().replace('"', '').split(',')
            data[row[0]] = row[1:]
    return data


def _read_raw(path):
    def _read(file):
        file_path = os.path.join(path, file + '.wav')
        return wavfile.read(file_path)[1]
    return _read


def _read_mfcc(path):
    path = path.replace('datasets', 'features/mfcc')

    def _read(file):
        file_path = os.path.join(path, file + '.npy')
        return np.load(file_path)
    return _read


def _read_mel(path, feature_length):
    path = path.replace('datasets', 'features/mel')

    def _read(file):
        file_path = os.path.join(path, file + '.npy')
        data = np.load(file_path)

        if data.shape[1] < feature_length:
            pad_width = [(0, 0), (0, feature_length - data.shape[1])]
            data = np.pad(data, pad_width, 'wrap')

        return data
    return _read


class DcaseAudioTagging(Sequence):

    def __init__(self, filename, batch_size=16, curated=None, features='mel',
                 feature_length=3000, path=None, shuffle=True, seed=None):
        if path is None:
            path = os.getcwd()

        self.rng = np.random.RandomState(seed)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.feature_length = feature_length

        if features == 'raw':
            self.read = _read_raw(path)
        elif features == 'mfcc':
            self.read = _read_mfcc(path)
        else:  # features == 'mel'
            self.read = _read_mel(path, self.feature_length)

        # read clip files
        file_path = os.path.join(path, filename)
        with open(file_path, 'r') as fp:
            self.filenames = [line.rstrip() for line in fp]

        # read labels and set up label mapping
        labels = {}
        if curated is None or curated:
            file_path = os.path.join(path, 'train_curated.csv')
            labels.update(_read_label_dict(file_path))
        if not curated:
            file_path = os.path.join(path, 'train_noisy.csv')
            labels.update(_read_label_dict(file_path))
        unique_labels = {l for label in labels.values() for l in label}
        self.num_classes = len(unique_labels)
        self.label_map = {l: i for i, l in enumerate(sorted(unique_labels))}
        self.labels = {k.rsplit('.wav')[0]: v for k, v in labels.items()}

    def __getitem__(self, index):
        start = index * self.batch_size
        batch_x = self.filenames[start:start + self.batch_size]
        labels = np.zeros((len(batch_x), self.num_classes))
        features = []
        for i, file in enumerate(batch_x):
            data = np.empty(0)
            try:
                data = self.read(os.path.join('train_curated', file))
            except FileNotFoundError:
                data = self.read(os.path.join('train_noisy', file))
            finally:
                features.append(data[:, :self.feature_length])

            tags = self.labels.get(file)
            indices = [self.label_map[tag] for tag in tags]
            labels[i, [indices]] = 1

        features = np.stack(features)[..., np.newaxis]
        return features, labels

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.filenames)


class DcaseAudioTesting(Sequence):
    def __init__(self, filenames, batch_size=16, path=None,
                 features='mel', feature_length=3000):
        if path is None:
            path = os.getcwd()

        self.filenames = filenames
        self.batch_size = batch_size
        self.feature_length = feature_length

        if features == 'raw':
            self.read = _read_raw(path)
        elif features == 'mfcc':
            self.read = _read_mfcc(path)
        else:  # features == 'mel'
            self.read = _read_mel(path, self.feature_length)

        # read labels and set up label mapping
        labels = {}
        file_path = os.path.join(path, 'train_curated.csv')
        labels.update(_read_label_dict(file_path))
        file_path = os.path.join(path, 'train_noisy.csv')
        labels.update(_read_label_dict(file_path))
        unique_labels = {l for label in labels.values() for l in label}
        self.num_classes = len(unique_labels)
        self.label_map = {l: i for i, l in enumerate(sorted(unique_labels))}

    def __getitem__(self, index):
        start = index * self.batch_size
        batch_x = self.filenames[start:start + self.batch_size]
        features = []
        for i, file in enumerate(batch_x):
            data = self.read(os.path.join('test', file))
            features.append(data[:, :self.feature_length])

        features = np.stack(features)[..., np.newaxis]
        return features

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))
