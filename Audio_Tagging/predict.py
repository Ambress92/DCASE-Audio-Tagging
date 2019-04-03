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
import keras


def opts_parser():
    descr = ("Computes predictions with a neural network.")
    parser = ArgumentParser(description=descr)
    parser.add_argument('modelfile', metavar='MODELFILE',
            type=str,
            help='File to load the learned weights from')
    parser.add_argument('--outfile', metavar='OUTFILE',
            type=str,
            help='File to save the prediction curves to (.npz/.pkl format)',
            default=None, required=False)
    config.prepare_argument_parser(parser)
    return parser

def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    modelfile = options.modelfile
    outfile = options.outfile


    # parse config
    if os.path.exists(vars):
        options.vars.insert(1, vars)
    cfg = config.from_parsed_arguments(options)

    # prepare dataset

    print("Preparing prediction function...")
    # instantiate neural network
    with open(modelfile.replace(".py", ".yaml"), 'r') as yaml_file:
        yaml_model = yaml_file.read()

    network = model_from_yaml(yaml_model)

    # load saved weights
    network.load_weights(modelfile.replace('.py', '.hd5'))

    # run prediction loop
    print("Predicting:")
    predictions = None


    # save predictions
    print("Saving predictions")
    with io.open(modelfile.replace('auto', outfile), 'wb') as f:
        np.save('predictions', predictions)

if __name__ == "__main__":
    main()
