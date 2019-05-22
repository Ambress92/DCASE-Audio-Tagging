#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Lambda, AveragePooling2D
import keras.backend as K

def shallow_cnn(data_format, num_classes):
    ini_filters = 64

    model = keras.models.Sequential()

    model.add(Conv2D(ini_filters, (5, 5), strides=2, activation='relu', padding='same', input_shape=data_format,
                     kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(Conv2D(ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(AveragePooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    model.add(
        Conv2D(2 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(
        Conv2D(2 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(AveragePooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    model.add(
        Conv2D(4 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(
        Conv2D(4 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(AveragePooling2D((2, 3), strides=(2, 3)))
    model.add(Dropout(0.3))

    model.add(
        Conv2D(8 * ini_filters, (1, 1), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(Dropout(0.5))

    # classification block
    model.add(Conv2D(num_classes, (1, 1), strides=1, activation='linear', padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(Lambda(lambda x: K.mean(x, axis=1)))
    model.add(Lambda(lambda x: K.max(x, axis=1)))
    # model.add(GlobalAveragePooling2D(data_format='channels_last'))
    model.add(Activation(activation='sigmoid'))

    print(model.summary())

    return model


def architecture(data_format, num_classes):
    """
    Instantiates a network model for a given dictionary of input/output
    tensor formats (dtype, shape) and a given configuration.
    """
    return shallow_cnn(data_format, num_classes)


