import librosa
import tqdm
import argparse
import os
from scipy.io import wavfile
import yaml
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-year', required=True)
args = parser.parse_args()

def dump_cqt_specs():
    files = os.listdir('../datasets/{}/audio_train'.format(args.year))

    for file in tqdm.tqdm(files, 'Extracting stft features'):
        sr, data = wavfile.read('../datasets/{}/audio_train/{}'.format(args.year, file))

        spec = librosa.cqt(data.astype(np.float), sr=sr)

        with open('../features/{}/cqt/{}.cqt'.format(args.year, file.split('.')[0]), 'w') as out_file:
            yaml.dump(spec, out_file)

def dump_mel_specs():
    n_fft = 1024
    sr = 32000
    n_mels = 128
    hop_length = 192
    fmax = None

    files = os.listdir('../datasets/{}/audio_train'.format(args.year))

    for file in tqdm.tqdm(files, 'Extracting mel spectrograms'):
        data, sr = librosa.load('../datasets/{}/audio_train/{}'.format(args.year, file), sr=sr, mono=True)


        stft = librosa.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=None, window='hann', center=True,
                            pad_mode='reflect')

        stft = np.abs(stft)
        stft = np.log10(stft+1)
        spec = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, fmax=fmax)


        with open('../features/{}/mel/{}.mel'.format(args.year, file.split('.')[0]), 'w') as out_file:
            yaml.dump(spec, out_file)

def dump_mfcc_features():
    files = os.listdir('../datasets/{}/audio_train'.format(args.year))

    for file in tqdm.tqdm(files, 'Extracting mfccs'):
        sr, data = wavfile.read('../datasets/{}/audio_train/{}'.format(args.year, file))

        mfcc = librosa.feature.mfcc(data.astype(np.float), sr, n_mfcc=40)
        #deltas = librosa.feature.delta(data)
        #delta_delta = librosa.feature.delta(data, order=2)

        with open('../features/{}/mfcc/{}.mfcc'.format(args.year, file.split('.')[0]), 'w') as out_file:
           yaml.dump(mfcc.T, out_file)


def main():
    # dump_mfcc_features()
    # dump_cqt_specs()
    dump_mel_specs()

if __name__ == '__main__':
    main()