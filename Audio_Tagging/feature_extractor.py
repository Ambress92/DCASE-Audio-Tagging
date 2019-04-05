import librosa
import tqdm
from argparse import ArgumentParser
import os
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--noisy', action='store_true')
parser.add_argument('--mfcc', action='store_true')
parser.add_argument('--cqt', action='store_true')
parser.add_argument('--centroids', action='store_true')
parser.add_argument('--melspectrogram', action='store_true')
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()

def plot_spectrogram(spectrogram, title):
    print("Spectrogram Shape:", spectrogram.shape)

    plt.figure()
    plt.clf()
    plt.subplots_adjust(right=0.98, left=0.1, bottom=0.1, top=0.99)
    plt.imshow(spectrogram, origin="lower", interpolation="nearest", cmap="viridis")
    plt.xlabel("%d frames" % spectrogram.shape[2])
    plt.ylabel("%d bins" % spectrogram.shape[1])
    plt.title(title)
    plt.colorbar()
    plt.show()

def normalize_features(features):
    features_mean = np.mean(features, axis=1)
    features_std = np.std(features, axis=1)
    normalized_features = (features - features_mean[:, np.newaxis]) / features_std[:, np.newaxis]
    return normalized_features

def dump_cqt_specs(dirname):
    """
    Computes constant Q-transform and dumps results into according directory.

    Parameters
    ----------
    use_train : boolean
        `True` if we want to compute cqt for training audio clips.
        `False` if we want to compute cqt to test clips.
    """

    files = os.listdir('../datasets/{}'.format(dirname))

    for file in tqdm.tqdm(files, 'Extracting stft features'):
        sr, data = wavfile.read('../datasets/{}/{}'.format(dirname, file))

        spec = librosa.cqt(data.astype(np.float), sr=sr)

        if args.plot:
            plot_spectrogram(spec[0], 'CQT Spectrogram')
            args.plot = False

        spec = normalize_features(spec)

        if args.plot:
            plot_spectrogram(spec[0], 'CQT Spectrogram Normalized')
            args.plot = False

        if not os.path.exists('../features/cqt/{}'.format(dirname)):
            os.makedirs('../features/cqt/{}'.format(dirname))

        np.save('../features/cqt/{}/{}'.format(dirname, file.split('.')[0]), spec)


def dump_mel_specs(dirname):
    """
    Computes Mel-scaled spectrogram and dumps results into according directory.

    Parameters
    ----------
    use_train : boolean
        `True` if we want to compute mel spectrograms for training audio clips.
        `False` if we want to compute mel spectrograms to test clips.
    """

    n_fft = 1024
    sr = 32000
    n_mels = 128
    hop_length = 192
    fmax = None

    files = os.listdir('../datasets/{}'.format(dirname))

    for file in tqdm.tqdm(files, 'Extracting mel spectrograms'):
        data, sr = librosa.load('../datasets/{}/{}/{}'.format(args.year, dirname, file), sr=sr, mono=True)


        stft = librosa.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=None, window='hann', center=True,
                            pad_mode='reflect')

        stft = np.abs(stft)
        stft = np.log10(stft+1)
        spec = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, fmax=fmax)


        if args.plot:
            plot_spectrogram(spec[0], 'Mel Spectrogram')
            args.plot = False

        spec = normalize_features(spec)


        if args.plot:
            plot_spectrogram(spec[0], 'Mel Spectrogram Normalized')
            args.plot = False

        if not os.path.exists('../features/cts/{}'.format(dirname)):
            os.makedirs('../features/cts/{}'.format(dirname))

        np.save('../features/mel/{}/{}'.format(dirname, file.split('.')[0]), spec)

def dump_mfcc_features(dirname):
    """
    Computes MFCCs and dumps features into according directory.

    Parameters
    ----------
    use_train : boolean
        `True` if we want to compute mfccs for training audio clips.
        `False` if we want to compute mfccs to test clips.
    """

    files = os.listdir('../datasets/{}'.format(dirname))

    for file in tqdm.tqdm(files, 'Extracting mfccs'):
        sr, data = wavfile.read('../datasets/{}/{}'.format(dirname, file))

        try:
            mfcc = librosa.feature.mfcc(data.astype(np.float), sr, n_mfcc=40)
        except:
            print('Extraction failed for file {}'.format(file))
        deltas = librosa.feature.delta(mfcc)
        delta_delta = librosa.feature.delta(mfcc, order=2)

        mfcc = np.vstack((mfcc, deltas, delta_delta))

        mfcc = normalize_features(mfcc.T)

        if not os.path.exists('../features/mfcc/{}'.format(dirname)):
            os.makedirs('../features/mfcc/{}'.format(dirname))

        np.save('../features/mfcc/{}/{}'.format(dirname, file.split('.')[0]), mfcc)

def dump_spectral_centroids(dirname):
    """
    Computes spectral centroid of audio samples and dumps results
    into according directory.

    Parameters
    ----------
    use_train : boolean
        `True` if we want to compute spectral centroids for training audio clips.
        `False` if we want to compute spectral centroids to test clips.
    """

    files = os.listdir('../datasets/{}'.format(dirname))

    for file in tqdm.tqdm(files, 'Extracting spectral centroids'):
        sr, data = wavfile.read('../datasets/{}/{}'.format(dirname, file))

        cts = librosa.feature.spectral_centroid(data.astype(np.float), sr)

        if not os.path.exists('../features/cts/{}'.format(dirname)):
            os.makedirs('../features/cts/{}'.format(dirname))

        np.save('../features/{}/centroids/{}'.format(dirname, file.split('.')[0]), cts.T)


def main():
    use_train = not args.test

    dirname = 'test'
    if use_train:
        if args.noisy:
            dirname = 'train_noisy'
        else:
            dirname = 'train_curated'

    if args.mfcc:
        if not os.path.exists('../features/mfcc'):
            os.makedirs('../features/mfcc/test')
            os.makedirs('../features/mfcc/train')
        dump_mfcc_features(dirname)
    if args.melspectrogram:
        if not os.path.exists('../features/mel'):
            os.makedirs('../features/mel/test')
            os.makedirs('../features/mel/train')
        dump_mel_specs(dirname)
    if args.cqt:
        if not os.path.exists('../features/cqt'):
            os.makedirs('../features/cqt/test')
            os.makedirs('../features/cqt/train')
        dump_cqt_specs(dirname)
    if args.centroids:
        if not os.path.exists('../features/centroids'):
            os.makedirs('../features/centroids/test')
            os.makedirs('../features/centroids/train')
        dump_spectral_centroids(dirname)

if __name__ == '__main__':
    main()
