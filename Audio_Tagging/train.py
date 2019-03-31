import argparse
from dataloader import get_verified_files_dict, load_verified_files, get_label_mapping, one_hot_encode
import numpy as np
import os
import utils
import keras

parser = argparse.ArgumentParser()
parser.add_argument('-year', required=True)
parser.add_argument('-features', required=True)
parser.add_argument('-clf', help='Classifier to use, by default RF is used', default='RF')
args = parser.parse_args()

def main():
    label_mapping, inv_label_mapping = get_label_mapping(args.year)

    print('Loading data...')
    data = load_verified_files(args.year, args.features)

    X = []
    y = []
    for x in data:
        label = x[1]
        x = x[0]
        for datapoint in x:
            X.append(datapoint)
            y.append(label_mapping[label])

    print('Load complete')
    X = np.asarray(X)
    y = np.asarray(y)

    print('Loading model')

    if not os.path.exists('models'):
        os.makedirs('models')

    if not os.path.exists('models/{}'.format(args.clf)):
        print('Modelfile does not exist')

    # import model from file
    selected_model = utils.import_model(args.clf)

    # instantiate neural network
    print("Preparing training function...")
    # network = selected_model.architecture(train_formats, cfg)

    # Add optimizer and compile model
    print("Compiling model ...")
    # optimizer = keras.optimizers.Adam(lr=cfg['lr'])
    # network.compile(optimizer=optimizer, loss=cfg["loss"], metrics=["acc"])


if __name__ == '__main__':
    main()
