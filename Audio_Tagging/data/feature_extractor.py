import os
from argparse import ArgumentParser

import librosa
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy.io import wavfile

parser = ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--noisy', action='store_true')
parser.add_argument('--mfcc', action='store_true')
parser.add_argument('--cqt', action='store_true')
parser.add_argument('--centroids', action='store_true')
parser.add_argument('--melspectrogram', action='store_true')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--spec_weighting', action='store_true')
args = parser.parse_args()


def plot_spectrogram(spectrogram, title, type):
    print("Spectrogram Shape:", spectrogram.shape)

    plt.figure()
    plt.subplots_adjust(right=0.98, left=0.1, bottom=0.1, top=0.99)
    if type == 'mel':
        plt.imshow(spectrogram, origin="lower", interpolation="nearest", cmap="viridis")
    else:
        plt.imshow(np.abs(spectrogram), origin='lower', interpolation='nearest', cmap='viridis')
    plt.xlabel("Time")
    plt.ylabel("%d bins" % spectrogram.shape[0])
    plt.title(title)
    # plt.colorbar()
    plt.tight_layout()
    plt.gcf().savefig('plots/{}.png'.format(title))


def normalize_features(features):
    normalized_features = (features - np.mean(features)) / np.std(features)
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

    n_bins = 84
    n_fft = 1024
    sr = 32000
    hop_length = 192
    files = os.listdir('../../datasets/{}'.format(dirname))

    for file in tqdm.tqdm(files, 'Extracting stft features'):
        # perform silence clipping
        aug_cmd = "norm -0.1 silence 1 0.025 0.15% norm -0.1 reverse silence 1 0.025 0.15% reverse"
        aug_audio_file = file.replace('.wav', '_clipped.wav')
        command = "sox %s %s %s" % (
        '../../datasets/{}/{}'.format(dirname, file), '../../datasets/{}/{}'.format(dirname, aug_audio_file), aug_cmd)
        os.system(command)

        assert os.path.exists('../../datasets/{}/{}'.format(dirname, aug_audio_file)), "SOX Problem ... clipped wav does not exist!"

        data, sr = librosa.load('../../datasets/{}/{}'.format(dirname, aug_audio_file), sr=sr, mono=True)
        if len(data) == 0:
            data, sr = librosa.load('../../datasets/{}/{}'.format(dirname, file), sr=sr, mono=True)

        cqt = librosa.core.cqt(data, sr=sr, hop_length=hop_length, n_bins=n_bins, pad_mode='reflect',
                                   fmin=librosa.note_to_hz('A1'))

        # keep only amplitudes
        cqt = np.abs(cqt)

        if args.spec_weighting:
            freqs = librosa.cqt_frequencies(n_bins, fmin=librosa.note_to_hz('A1'))
            cqt = librosa.perceptual_weighting(cqt ** 2, freqs, ref=np.max)


        if args.plot:
            sr, orig_data = wavfile.read('../../datasets/{}/{}'.format(dirname, file))
            spec_orig = librosa.cqt(orig_data.astype(np.float), sr=sr, n_bins=84)
            spec_orig = np.abs(spec_orig)

            plot_spectrogram(spec_orig, 'Original CQT Spectrogram', 'cqt')
            plot_spectrogram(spec, 'CQT Spectrogram after silence clipping', 'cqt')

        spec = normalize_features(cqt)
        os.remove('../../datasets/{}/{}'.format(dirname, aug_audio_file))

        if args.plot:
            plot_spectrogram(spec, 'CQT Spectrogram Normalized', 'cqt')
            args.plot = False

        if not os.path.exists('../../features/cqt/{}'.format(dirname)):
            os.makedirs('../../features/cqt/{}'.format(dirname))
        if not os.path.exists('../../features/cqt_weighted/{}'.format(dirname)):
            os.makedirs('../../features/cqt_weighted/{}'.format(dirname))

        if args.spec_weighting:
            np.save('../../features/cqt_weighted/{}/{}'.format(dirname, file.split('.')[0]), cqt)
        else:
            np.save('../../features/cqt/{}/{}'.format(dirname, file.split('.')[0]), cqt)


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

    files = os.listdir('../../datasets/{}'.format(dirname))

    for file in tqdm.tqdm(files, 'Extracting mel spectrograms'):
        # perform silence clipping
        aug_cmd = "norm -0.1 silence 1 0.025 0.15% norm -0.1 reverse silence 1 0.025 0.15% reverse"
        aug_audio_file = file.replace('.wav', '_clipped.wav')
        command = "sox %s %s %s" % ('../../datasets/{}/{}'.format(dirname, file), '../../datasets/{}/{}'.format(dirname, aug_audio_file), aug_cmd)
        os.system(command)

        assert os.path.exists('../../datasets/{}/{}'.format(dirname, aug_audio_file)), "SOX Problem ... clipped wav does not exist!"

        data, sr = librosa.load('../../datasets/{}/{}'.format(dirname, aug_audio_file), sr=sr, mono=True)

        if len(data) == 0:
            data, sr = librosa.load('../../datasets/{}/{}'.format(dirname, file), sr=sr, mono=True)

        stft = librosa.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=None, window='hann', center=True,
                                pad_mode='reflect')

        # keep only amplitudes of spectrograms
        stft = np.abs(stft)

        if args.spec_weighting:
            freqs = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)
            stft = librosa.perceptual_weighting(stft**2, freqs, ref=1.0, amin=1e-10, top_db=99.0)
        else:
            stft = np.log10(stft+1)

        # apply mel filterbank
        spec = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, fmax=fmax)


        if args.plot:
            orig_data, sr = librosa.load('../../datasets/{}/{}'.format(dirname, file), sr=sr, mono=True)
            stft_orig = librosa.stft(orig_data, n_fft=n_fft, hop_length=hop_length, win_length=None, window='hann', center=True,
                                pad_mode='reflect')

            stft_orig = np.abs(stft_orig)
            stft_orig = np.log10(stft_orig + 1)
            spec_orig= librosa.feature.melspectrogram(S=stft_orig, sr=sr, n_mels=n_mels, fmax=fmax)

            plot_spectrogram(spec_orig, 'Original Mel Spectrogram', 'mel')
            plot_spectrogram(spec, 'Mel Spectrogram after silence clipping', 'mel')

        spec = normalize_features(spec)
        os.remove('../../datasets/{}/{}'.format(dirname, aug_audio_file))


        if args.plot:
            plot_spectrogram(spec, 'Mel Spectrogram Normalized', 'mel')
            args.plot = False

        if not os.path.exists('../../features/mel/{}'.format(dirname)):
            os.makedirs('../../features/mel/{}'.format(dirname))
        if not os.path.exists('../../features/mel_weighted/{}'.format(dirname)):
            os.makedirs('../../features/mel_weighted/{}'.format(dirname))

        if args.spec_weighting:
            np.save('../../features/mel_weighted/{}/{}'.format(dirname, file.split('.')[0]), spec)
        else:
            np.save('../../features/mel/{}/{}'.format(dirname, file.split('.')[0]), spec)


def dump_mfcc_features(dirname):
    """
    Computes MFCCs and dumps features into according directory.
    Parameters
    ----------
    use_train : boolean
        `True` if we want to compute mfccs for training audio clips.
        `False` if we want to compute mfccs to test clips.
    """

    files = os.listdir('../../datasets/{}'.format(dirname))

    for file in tqdm.tqdm(files, 'Extracting mfccs'):
        sr, data = wavfile.read('../../datasets/{}/{}'.format(dirname, file))

        try:
            mfcc = librosa.feature.mfcc(data.astype(np.float), sr, n_mfcc=40)
        except:
            print('Extraction failed for file {}'.format(file))

        #deltas = librosa.feature.delta(mfcc)
        #delta_delta = librosa.feature.delta(mfcc, order=2)

        #mfcc = np.vstack((mfcc, deltas, delta_delta))
        chunk = int(mfcc.shape[1]/10)
        mfcc = mfcc[:, :chunk]
        mfcc = normalize_features(mfcc.T)

        if not os.path.exists('../../features/mfcc/{}'.format(dirname)):
            os.makedirs('../../features/mfcc/{}'.format(dirname))

        np.save('../../features/mfcc/{}/{}'.format(dirname, file.split('.')[0]), mfcc)


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

    files = os.listdir('../../datasets/{}'.format(dirname))

    for file in tqdm.tqdm(files, 'Extracting spectral centroids'):
        sr, data = wavfile.read('../../datasets/{}/{}'.format(dirname, file))

        cts = librosa.feature.spectral_centroid(data.astype(np.float), sr)

        if not os.path.exists('../../features/cts/{}'.format(dirname)):
            os.makedirs('../../features/cts/{}'.format(dirname))

        np.save('../../features/{}/centroids/{}'.format(dirname, file.split('.')[0]), cts.T)


def main():
    use_train = not args.test

    dirname = 'test'
    if use_train:
        if args.noisy:
            dirname = 'train_noisy'
        else:
            dirname = 'train_curated'

    if args.mfcc:
        if not os.path.exists('../../features/mfcc'):
            os.makedirs('../../features/mfcc/{}'.format(dirname))
        dump_mfcc_features(dirname)
    if args.melspectrogram:
        if not os.path.exists('../../features/mel'):
            os.makedirs('../../features/mel/{}'.format(dirname))
        dump_mel_specs(dirname)
    if args.cqt:
        if not os.path.exists('../../features/cqt'):
            os.makedirs('../../features/cqt/{}'.format(dirname))
        dump_cqt_specs(dirname)
    if args.centroids:
        if not os.path.exists('../../features/centroids'):
            os.makedirs('../../features/centroids/{}'.format(dirname))
        dump_spectral_centroids(dirname)


if __name__ == '__main__':
    main()