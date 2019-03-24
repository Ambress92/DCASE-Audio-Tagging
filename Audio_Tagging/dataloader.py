from scipy.io import wavfile
import tqdm
import numpy as np
import yaml

def get_verified_files_dict(year):
    with open('../datasets/{}/train.csv'.format(year), 'r') as in_file:
        data_config = in_file.readlines()
        data_config = data_config[1:]

    verified_files_dict = {line.split(',')[0]: line.split(',')[1] for line in data_config if line.split(',')[2].rstrip() == '0'}

    return verified_files_dict

def get_unverified_files_dict(year):
    with open('../datasets/{}/train.csv'.format(year), 'r') as in_file:
        data_config = in_file.readlines()
        data_config = data_config[1:]

    unverified_files_dict = {line.split(',')[0]: line.split(',')[1] for line in data_config if line.split(',')[2].rstrip() == '1'}

    return unverified_files_dict

def load_verified_files(year, load_features=False):
    verified_files_dict = get_verified_files_dict(year)

    # load verified audio clips
    verified_files = []
    for file, label in tqdm.tqdm(zip(verified_files_dict.keys(), verified_files_dict.values()), 'Loading verified clips'):
        if not load_features:
            _, data = wavfile.read('../datasets/{}/audio_train/{}'.format(year, file))
        else:
            with open('../features/{}/mfcc/{}'.format(year, file), 'r') as in_file:
                data = yaml.load(in_file)

        verified_files.append((data, label))

    return verified_files

def load_unverified_files(year, load_features=False):
    unverified_files_dict = get_unverified_files_dict(year)

    # load verified audio clips
    unverified_files = []
    for file, label in tqdm.tqdm(zip(unverified_files_dict.keys(), unverified_files_dict.values()),
                                 'Loading verified clips'):
        if not load_features:
            _, data = wavfile.read('../datasets/{}/audio_train/{}'.format(year, file))
        else:
            with open('../features/{}/mfcc/{}'.format(year, file), 'r') as in_file:
                data = yaml.load(in_file)

        unverified_files.append((data, label))

    return unverified_files

def get_label_mapping():
    verified_files_dict = get_verified_files_dict()
    labels = np.unique(list(verified_files_dict.values()))
    label_mapping = {label: index for index, label in enumerate(labels)}
    inv_label_mapping = {v: k for k, v in zip(label_mapping.keys(), label_mapping.values())}
    return label_mapping, inv_label_mapping
