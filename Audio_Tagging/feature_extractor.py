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
    files = os.listdir('../datasets/{}/audio_train'.format(args.year))

    for file in tqdm.tqdm(files, 'Extracting mel spectrograms'):
        sr, data = wavfile.read('../datasets/{}/audio_train/{}'.format(args.year, file))

        spec = librosa.feature.melspectrogram(data.astype(np.float), sr)

        with open('../features/{}/mel/{}.mel'.format(args.year, file.split('.')[0]), 'w') as out_file:
            yaml.dump(spec, out_file)

def dump_mfcc_features():
    files = os.listdir('../datasets/{}/audio_train'.format(args.year))

    for file in tqdm.tqdm(files, 'Extracting mfccs'):
        sr, data = wavfile.read('../datasets/{}/audio_train/{}'.format(args.year, file))

        mfcc = librosa.feature.mfcc(data, sr, n_mfcc=40)
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