import librosa
import tqdm
import argparse
import os
from scipy.io import wavfile
import yaml
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-year', required=True)
parser.add_argument('--test', action='store_true')
parser.add_argument('--mfcc', action='store_true')
parser.add_argument('--cqt', action='store_true')
parser.add_argument('--centroids', action='store_true')
parser.add_argument('--melspectrogram', action='store_true')
args = parser.parse_args()

def dump_cqt_specs(use_train):
    """
    Computes constant Q-transform and dumps results into according directory.

    Parameters
    ----------
    use_train : boolean
        `True` if we want to compute cqt for training audio clips.
        `False` if we want to compute cqt to test clips.
    """

    dirname = 'audio_test'
    if use_train:
        dirname = 'audio_train'

    files = os.listdir('../datasets/{}/{}'.format(args.year, dirname))

    for file in tqdm.tqdm(files, 'Extracting stft features'):
        sr, data = wavfile.read('../datasets/{}/{}/{}'.format(args.year, dirname, file))

        spec = librosa.cqt(data.astype(np.float), sr=sr)

        with open('../features/{}/cqt/{}/{}.cqt'.format(args.year, dirname, file.split('.')[0]), 'w') as out_file:
            yaml.dump(spec, out_file)

def dump_mel_specs(use_train):
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

    dirname = 'audio_test'
    if use_train:
        dirname = 'audio_train'

    files = os.listdir('../datasets/{}/{}'.format(args.year, dirname))

    for file in tqdm.tqdm(files, 'Extracting mel spectrograms'):
        data, sr = librosa.load('../datasets/{}/{}/{}'.format(args.year, dirname, file), sr=sr, mono=True)


        stft = librosa.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=None, window='hann', center=True,
                            pad_mode='reflect')

        stft = np.abs(stft)
        stft = np.log10(stft+1)
        spec = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, fmax=fmax)


        with open('../features/{}/mel/{}/{}.mel'.format(args.year, dirname, file.split('.')[0]), 'w') as out_file:
            yaml.dump(spec, out_file)

def dump_mfcc_features(use_train):
    """
    Computes MFCCs and dumps features into according directory.

    Parameters
    ----------
    use_train : boolean
        `True` if we want to compute mfccs for training audio clips.
        `False` if we want to compute mfccs to test clips.
    """

    dirname = 'audio_test'
    if use_train:
        dirname = 'audio_train'

    files = os.listdir('../datasets/{}/{}'.format(args.year, dirname))

    for file in tqdm.tqdm(files, 'Extracting mfccs'):
        sr, data = wavfile.read('../datasets/{}/{}/{}'.format(args.year, dirname, file))

        try:
            mfcc = librosa.feature.mfcc(data.astype(np.float), sr, n_mfcc=40)
        except:
            print('Extraction failed for file {}'.format(file))
        #deltas = librosa.feature.delta(data)
        #delta_delta = librosa.feature.delta(data, order=2)

        with open('../features/{}/mfcc/{}/{}.mfcc'.format(args.year, dirname, file.split('.')[0]), 'w') as out_file:
           yaml.dump(mfcc.T, out_file)

def dump_spectral_centroids(use_train):
    """
    Computes spectral centroid of audio samples and dumps results
    into according directory.

    Parameters
    ----------
    use_train : boolean
        `True` if we want to compute spectral centroids for training audio clips.
        `False` if we want to compute spectral centroids to test clips.
    """

    dirname = 'audio_test'
    if use_train:
        dirname = 'audio_train'

    files = os.listdir('../datasets/{}/{}'.format(args.year, dirname))

    for file in tqdm.tqdm(files, 'Extracting spectral centroids'):
        sr, data = wavfile.read('../datasets/{}/{}/{}'.format(args.year, dirname, file))

        cts = librosa.feature.spectral_centroid(data.astype(np.float), sr)

        with open('../features/{}/centroids/{}/{}.cts'.format(args.year, dirname, file.split('.')[0]), 'w') as out_file:
            yaml.dump(cts, out_file)



def main():
    use_train = not args.test

    if args.mfcc:
        if not os.path.exists('../features/{}/mfcc'.format(args.year)):
            os.makedirs('../features/{}/mfcc/audio_train'.format(args.year))
            os.makedirs('../features/{}/mfcc/audio_test'.format(args.year))
        dump_mfcc_features(use_train)
    if args.melspectrogram:
        if not os.path.exists('../features/{}/mel'.format(args.year)):
            os.makedirs('../features/{}/mel/audio_train'.format(args.year))
            os.makedirs('../features/{}/mel/audio_test'.format(args.year))
        dump_mel_specs(use_train)
    if args.cqt:
        if not os.path.exists('../features/{}/cqt'.format(args.year)):
            os.makedirs('../features/{}/cqt/audio_train'.format(args.year))
            os.makedirs('../features/{}/cqt/audio_test'.format(args.year))
        dump_cqt_specs(use_train)
    if args.centroids:
        if not os.path.exists('../features/{}/centroids'.format(args.year)):
            os.makedirs('../features/{}/centroids/audio_train'.format(args.year))
            os.makedirs('../features/{}/centroids/audio_test'.format(args.year))
        dump_spectral_centroids(use_train)

if __name__ == '__main__':
    main()
