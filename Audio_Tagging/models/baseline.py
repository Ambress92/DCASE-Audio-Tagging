#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model definition for visibility condition classification experiment.
Provides architecture().

Author: Jan Schlüter, Fabian Paischer and Matthias Dorfer
"""

import keras
from keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D, Activation, MaxPooling2D, Dropout


def baseline(data_format, num_classes):

    ini_filters = 64

    model = keras.models.Sequential()

    model.add(Conv2D(ini_filters, (5, 5), strides=2, activation='relu', padding='same', input_shape=data_format))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(2 * ini_filters, (3, 3), strides=1, activation='relu', padding='same'))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(4 * ini_filters, (3, 3), strides=1, activation='relu', padding='same'))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(Dropout(0.3))

    model.add(Conv2D(8 * ini_filters, (3, 3), strides=1, activation='relu', padding='same'))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(MaxPooling2D((1, 2), strides=(1, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(8 * ini_filters, (3, 3), strides=1, activation='relu', padding='same'))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(MaxPooling2D((1, 2), strides=(1, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(8 * ini_filters, (3, 3), strides=1, activation='relu', padding='same'))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(Dropout(0.5))

    # classification block
    model.add(Conv2D(num_classes, (1, 1), strides=1, activation='relu', padding='same'))
    model.add(GlobalAveragePooling2D(data_format='channels_last'))
    model.add(Activation(activation='sigmoid'))

    print(model.summary())

    return model


def architecture(data_format, cfg):
    """
    Instantiates a network model for a given dictionary of input/output
    tensor formats (dtype, shape) and a given configuration.
    """
    return baseline(data_format, cfg)


