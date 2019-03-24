import librosa
import tqdm
import argparse
import os
from scipy.io import wavfile
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('-year', required=True)
args = parser.parse_args()

def dump_stft_specs():
    files = os.listdir('../datasets/{}/audio_train'.format(args.year))

    for file in tqdm.tqdm(files, 'Extracting mfccs'):
        sr, data = wavfile.read('../datasets/{}/audio_train/{}'.format(args.year, file))

        spec = librosa.stft(data)

        with open('../features/{}/mfcc/{}.mfcc'.format(args.year, file.split('.')[0]), 'w') as out_file:
            yaml.dump(spec, out_file)

def dump_mel_specs():
    files = os.listdir('../datasets/{}/audio_train'.format(args.year))

    for file in tqdm.tqdm(files, 'Extracting mfccs'):
        sr, data = wavfile.read('../datasets/{}/audio_train/{}'.format(args.year, file))

        spec = librosa.feature.melspectrogram(data, sr)

        with open('../features/{}/mfcc/{}.mfcc'.format(args.year, file.split('.')[0]), 'w') as out_file:
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
    dump_mfcc_features()

if __name__ == '__main__':
    main()