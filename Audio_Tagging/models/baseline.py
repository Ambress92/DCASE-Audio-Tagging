#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model definition for visibility condition classification experiment.
Provides architecture().

Author: Jan Schlüter, Fabian Paischer and Matthias Dorfer
"""

import keras
from keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D, Activation, MaxPooling2D, Dropout, AveragePooling2D, Input, Lambda, Flatten, Dense
import keras.backend as K


def baseline(data_format, num_classes):
    ini_filters = 64

    inputs = Input(data_format)

    model = Conv2D(ini_filters, (3, 3), strides=1, activation='relu', padding='same', input_shape=data_format,
                   kernel_initializer='he_normal', use_bias=False)(inputs)
    model = BatchNormalization(momentum=0.9, axis=-1)(model)
    model = AveragePooling2D((2, 2), strides=(2, 2))(model)
    model = Conv2D(ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
                   use_bias=False)(model)
    model = BatchNormalization(momentum=0.9, axis=-1)(model)
    model = AveragePooling2D((2, 2), strides=(2, 2))(model)

    model = Conv2D(2 * ini_filters, (3, 3), strides=1, activation='relu', padding='same',
                   kernel_initializer='he_normal', use_bias=False)(model)
    model = BatchNormalization(momentum=0.9, axis=-1)(model)
    model = AveragePooling2D((2, 2), strides=(2, 2))(model)
    model = Conv2D(2 * ini_filters, (3, 3), strides=1, activation='relu', padding='same',
                   kernel_initializer='he_normal', use_bias=False)(model)
    model = BatchNormalization(momentum=0.9, axis=-1)(model)
    model = AveragePooling2D((2, 2), strides=(2, 2))(model)

    model = Conv2D(4 * ini_filters, (3, 3), strides=1, activation='relu', padding='same',
                   kernel_initializer='he_normal', use_bias=False)(model)
    model = BatchNormalization(momentum=0.9, axis=-1)(model)
    model = AveragePooling2D((2, 2), strides=(2, 2))(model)
    model = Conv2D(4 * ini_filters, (3, 3), strides=1, activation='relu', padding='same',
                   kernel_initializer='he_normal', use_bias=False)(model)
    model = BatchNormalization(momentum=0.9, axis=-1)(model)
    model = AveragePooling2D((2, 2), strides=(2, 2))(model)

    model = Conv2D(8 * ini_filters, (3, 3), strides=1, activation='relu', padding='same',
                   kernel_initializer='he_normal', use_bias=False)(model)
    model = BatchNormalization(momentum=0.9, axis=-1)(model)
    model = AveragePooling2D((2, 2), strides=(2, 2))(model)
    model = Conv2D(8 * ini_filters, (3, 3), strides=1, activation='relu', padding='same',
                   kernel_initializer='he_normal', use_bias=False)(model)
    model = BatchNormalization(momentum=0.9, axis=-1)(model)
    model = AveragePooling2D((1, 2), strides=(1, 2))(model)

    model = Lambda(lambda x: K.mean(x, axis=1)[:, None, :, :])(model)
    model = Lambda(lambda x: K.max(x, axis=2))(model)
    model = Flatten()(model)
    output = Dense(num_classes, activation='softmax')(model)

    model = keras.models.Model(inputs=[inputs], outputs=[output])
    print(model.summary())

    return model


def architecture(data_format, cfg):
    """
    Instantiates a network model for a given dictionary of input/output
    tensor formats (dtype, shape) and a given configuration.
    """
    return baseline(data_format, cfg)


