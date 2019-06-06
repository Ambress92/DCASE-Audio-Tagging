#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Lambda, Flatten, Dense
import keras.backend as K


def cpjku_2018_cnn(data_format, num_classes):
    ini_filters = 64

    model = keras.models.Sequential()

    model.add(Conv2D(ini_filters, (5, 5), strides=2, activation='relu', padding='same', input_shape=data_format,
                     kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(Conv2D(ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
                     use_bias=False))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    model.add(
        Conv2D(2 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(
        Conv2D(2 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    model.add(
        Conv2D(4 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(Dropout(0.3))
    model.add(
        Conv2D(4 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(Dropout(0.3))
    model.add(
        Conv2D(6 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(Dropout(0.3))
    model.add(
        Conv2D(6 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    model.add(
        Conv2D(8 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(
        Conv2D(8 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(MaxPooling2D((1, 2), strides=(1, 2)))
    model.add(Dropout(0.3))

    model.add(
        Conv2D(8 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(
        Conv2D(8 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(MaxPooling2D((1, 2), strides=(1, 2)))
    model.add(Dropout(0.3))

    model.add(
        Conv2D(8 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(Dropout(0.5))
    model.add(
        Conv2D(8 * ini_filters, (1, 1), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False))
    model.add(BatchNormalization(momentum=0.9, axis=-1))
    model.add(Dropout(0.5))

    # classification block
    model.add(Lambda(lambda x: K.mean(x, axis=1)[:, None, :, :]))
    model.add(Lambda(lambda x: K.max(x, axis=2)))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    print(model.summary())

    return model


def architecture(data_format, num_classes):
    """
    Instantiates a network model for a given dictionary of input/output
    tensor formats (dtype, shape) and a given configuration.
    """
    return cpjku_2018_cnn(data_format, num_classes)

