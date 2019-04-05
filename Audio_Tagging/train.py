import argparse
import dataloader
import numpy as np
import os
import utils
import keras
from argparse import ArgumentParser
import config
import tqdm

def opts_parser():
    descr = "Trains a neural network."
    parser = ArgumentParser(description=descr)
    parser.add_argument('modelfile', metavar='MODELFILE',
            type=str,
            help='File to save the learned weights to')
    config.prepare_argument_parser(parser)
    return parser

def save_model(modelfile, network, cfg):
    """
    Saves the learned weights to `filename`, and the corresponding
    configuration to ``os.path.splitext(filename)[0] + '.vars'``.
    """
    config.write_config_file(modelfile + '_auto.vars', cfg)
    network_yaml = network.to_yaml()
    with open(modelfile.replace('.py', ".yaml"), 'w') as yaml_file:
        yaml_file.write(network_yaml)

def main():
    label_mapping, inv_label_mapping = dataloader.get_label_mapping()
    parser = opts_parser()
    options = parser.parse_args()
    modelfile = options.modelfile
    cfg = config.from_parsed_arguments(options)
    keras.backend.set_image_data_format('channels_first')

    verified_files_dict = dataloader.get_verified_files_dict()
    noisy_files_dict = dataloader.get_unverified_files_dict()
    total_files_dict = dict(verified_files_dict, **noisy_files_dict)

    print('Loading data...')
    for fold in range(1,5):
        # add data loading here
        with open('../datasets/fold{}_train'.format(fold), 'r') as in_file:
            train_files = in_file.readlines()
        with open('../datasets/fold{}_eval'.format(fold), 'r') as in_file:
            eval_files = in_file.readlines()

        print('Loading training data')
        X_train = dataloader.load_features(train_files)
        X_test = dataloader.load_features(eval_files)
        y_train = [dataloader.one_hot_encode(label_mapping(total_files_dict[file]), cfg['num_classes']) for file in train_files]
        y_test = [dataloader.one_hot_encode(label_mapping(total_files_dict[file]), cfg['num_classes']) for file in eval_files]
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        print('Loading model')
        # import model from file
        selected_model = utils.import_model(modelfile)

        # instantiate neural network
        print("Preparing training function...")

        train_formats = (cfg['batch_size'], 1, X_train[0].shape[0], X_train[0].shape[1])
        network = selected_model.architecture(train_formats, cfg)

        # Add optimizer and compile model
        print("Compiling model ...")
        optimizer = keras.optimizers.Adam(lr=cfg['lr'])
        network.compile(optimizer=optimizer, loss=cfg["loss"], metrics=["acc"])

        print("Preserving architecture and configuration ..")
        save_model(modelfile.replace('.py', ''), network, cfg)

        # Add batch creator, and training procedure
        val_loss = []
        val_acc = []
        train_loss = []
        train_acc = []
        epochs_without_decrase = 0
        f_scores = []

        # run training loop
        print("Training:")
        for epoch in range(cfg['epochs']):

            epoch_train_loss = []
            epoch_train_acc = []
            batch_val_loss = []
            batch_val_acc = []

            for _ in tqdm.trange(
                    cfg['epochsize'],
                    desc='Epoch %d/%d:' % (epoch + 1, cfg['epochs'])):
                pass

if __name__ == '__main__':
    main()
