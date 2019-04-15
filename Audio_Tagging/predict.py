#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Computes predictions with a neural network.

For usage information, call with --help.

Authors: Jan Schlüter, Fabian Paischer, Matthias Dorfer
"""

from __future__ import print_function

import io
import os
from argparse import ArgumentParser

import numpy as np
import tqdm

import config
from keras.models import model_from_yaml
from dataloader import load_batches, load_features, load_test_features


def opts_parser():
    descr = ("Computes predictions with a neural network.")
    parser = ArgumentParser(description=descr)
    parser.add_argument('modelfile', metavar='MODELFILE',
            type=str,
            help='File to load the learned weights from')
    parser.add_argument('--outfile', metavar='OUTFILE',
            type=str,
            help='File to save the prediction curves to (.npy/.pkl format)',
            default=None, required=False)
    parser.add_argument('--filelist', type=str, help='filelist to predict the labels for', default='validation',
                        required=False)
    parser.add_argument('--features', default='mel', type=str, help='features to predict on', required=False)
    config.prepare_argument_parser(parser)
    return parser

def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    modelfile = options.modelfile

    # parse config
    cfg = config.from_parsed_arguments(options)


    print("Preparing prediction function...")
    # instantiate neural network
    with open(modelfile, 'r') as yaml_file:
        yaml_model = yaml_file.read()

    network = model_from_yaml(yaml_model)

    # load saved weights
    network.load_weights(modelfile.replace('.yaml', '.hd5'))

    # run prediction loop
    print("Predicting:")
    fold = int(modelfile.split('_')[2].split('.')[0][-1])
    if options.filelist == 'validation':
        with open('../datasets/cv/fold{}_curated_eval'.format(fold), 'r') as in_file:
            filelist = in_file.readlines()
        batches = load_batches(filelist, cfg['batchsize'])
    else:
        filelist = os.listdir('../features/{}/test'.format(options.features))
        filelist = [file.replace('.npy', '') for file in filelist]
        batches = load_batches(filelist, cfg['batchsize'])

    predictions = []
    truth = []
    for batch in tqdm.tqdm(batches, desc='Batch'):
        if options.filelist == 'validation':
            X, y = load_features(batch, features=options.features, num_classes=cfg['num_classes'])
            X = X[:, :, :, np.newaxis]
        else:
            X = load_test_features(batch, features=options.features)

        preds = network.predict(x=X, batch_size=cfg['batchsize'], verbose=0)
        predictions.extend(preds)
        if options.filelist == 'validation':
            truth.extend(y)

    # save predictions
    print("Saving predictions")
    if options.filelist == 'validation':
        np.save('predictions/{}_predictions_fold{}'.format(modelfile.split('/')[1].replace('.yaml', ''), fold), np.asarray(predictions))
        np.save('predictions/{}_truth_fold{}'.format(modelfile.split('/')[1].replace('.yaml', ''), fold), np.asarray(truth))
    else:
        np.save('predictions/{}_predictions_test'.format(modelfile.split('/')[1].replace('.yaml', ''), fold),
                np.asarray(predictions))

if __name__ == "__main__":
    main()
