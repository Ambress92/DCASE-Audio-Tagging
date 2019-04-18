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
import re


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
    fold = int(re.match(f'.*fold([0-9]).*', modelfile).group(1))
    if options.filelist == 'validation':
        with open('../datasets/cv/fold{}_curated_eval'.format(fold), 'r') as in_file:
            filelist = in_file.readlines()
        batches = load_batches(filelist, cfg['batchsize'])
    else:
        filelist = os.listdir('../features/{}/test'.format(options.features))
        filelist = [file.replace('.npy', '') for file in filelist]
        batches = load_batches(filelist, cfg['batchsize'], test=True)

    predictions = []
    truth = []

    if options.filelist == 'test':

        for X, batch in tqdm.tqdm(batches):
            preds = network.predict(x=X, batch_size=cfg['batchsize'], verbose=0)
            predictions.extend(np.average(preds, axis=0))

    else:
        for X, y  in tqdm.tqdm(batches, desc='Batch'):
            preds = network.predict(x=X, batch_size=cfg['batchsize'], verbose=0)
            predictions.append(np.average(preds, axis=0))
            truth.append(np.average(y, axis=0))

    pred_dict = {}
    truth_dict = {}
    for i, file in enumerate(filelist):
        pred_dict[file] = predictions[i]
        if options.filelist == 'validation':
            truth_dict[file] = truth[i]

    # save predictions
    print("Saving predictions")
    if options.filelist == 'validation':
        np.save('predictions/{}_predictions_fold{}'.format(modelfile.split('/')[1].replace('.yaml', ''), fold), pred_dict)
        np.save('predictions/{}_truth_fold{}'.format(modelfile.split('/')[1].replace('.yaml', ''), fold), truth_dict)
    else:
        np.save('predictions/{}_predictions_test'.format(modelfile.split('/')[1].replace('.yaml', ''), fold),
                np.asarray(pred_dict))


if __name__ == "__main__":
    main()
