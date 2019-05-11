#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
from argparse import ArgumentParser
import numpy as np
np.random.seed(101)
import tqdm
import config
from keras.models import model_from_yaml
from dataloader import load_batches, load_features, load_test_features
import re
from keras.models import load_model


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
                        required=False, choices=['validation', 'test'])
    parser.add_argument('--level', type=str, help='predict on file level or on frame level', default='file',
                        choices=['file', 'frame'])
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
    if '.yaml' in modelfile:
        with open(modelfile, 'r') as yaml_file:
            yaml_model = yaml_file.read()
        network = model_from_yaml(yaml_model)
        network.load_weights(modelfile.replace('.yaml', '.hd5'))
    else:
        network = load_model(modelfile)

    # run prediction loop
    print("Predicting:")
    fold = int(re.match(f'.*fold([0-9]).*', modelfile).group(1))
    if options.filelist == 'validation':
        with open('../datasets/cv/fold{}_curated_eval'.format(fold), 'r') as in_file:
            filelist = in_file.readlines()
        batches = load_batches(filelist, cfg['batchsize'], test=False, augment=False, already_saved=True)
    else:
        filelist = os.listdir('../features/{}/test'.format(cfg['features']))
        filelist = [file.replace('.npy', '') for file in filelist]
        batches = load_batches(filelist, cfg['batchsize'], test=True)

    predictions = []
    truth = []

    if options.filelist == 'test':

        for X in tqdm.tqdm(batches):
            preds = network.predict(x=X, batch_size=cfg['batchsize'], verbose=0)
            for i in range(0, len(X), cfg['feature_width']):
                if options.level == 'file':
                    predictions.append(np.average(preds[i:i+cfg['feature_width']], axis=0))
                else:
                    predictions.append(preds[i:i+cfg['feature_width']])

    else:
        for X, y  in tqdm.tqdm(batches, desc='Batch'):
            preds = network.predict(x=X, batch_size=cfg['batchsize'], verbose=0)
            for i in range(0, len(X), cfg['feature_width']):
                if options.level == 'file':
                    predictions.append(np.average(preds[i:i+cfg['feature_width']], axis=0))
                else:
                    predictions.append(preds[i:i+cfg['feature_width']])
                truth.append(np.average(y[i:i + cfg['feature_width']], axis=0))


    pred_dict = {}
    truth_dict = {}
    for i, file in enumerate(filelist):
        pred_dict[file] = predictions[i]
        if options.filelist == 'validation':
            truth_dict[file] = truth[i]

    # save predictions
    print("Saving predictions")
    if not os.path.exists('predictions/{}'.format(cfg['features'])):
        os.makedirs('predictions/{}'.format(cfg['features']))

    model_extension = modelfile[modelfile.index('.'):]
    if options.filelist == 'validation':
        np.save('predictions/{}/{}_predictions_{}_fold{}'.format(cfg['features'], modelfile.split('/')[2].replace(model_extension, ''), options.level,
                                                              fold), pred_dict)
        if not os.path.exists('predictions/{}/truth_{}_fold{}'.format(cfg['features'], options.level, fold)):
            np.save('predictions/{}/truth_{}_fold{}'.format(cfg['features'], options.level, fold), truth_dict)
    else:
        np.save('predictions/{}/{}_predictions_{}_test'.format(cfg['features'], modelfile.split('/')[2].replace(model_extension, ''), options.level),
                np.asarray(pred_dict))


if __name__ == "__main__":
    main()
