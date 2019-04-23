#!/usr/bin/env python
# -*- coding: utf-8 -*-
import keras
from keras.layers import Conv2D, GlobalMaxPooling2D, Dense, MaxPooling2D


def get_model(data_format, num_classes):

    model = keras.models.Sequential()

    # convolutional and max pool layers
    model.add(Conv2D(filters=100, kernel_size=(7, 7), strides=1, activation='relu', padding='same', input_shape=data_format))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Conv2D(filters=150, kernel_size=(5, 5), strides=1, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Conv2D(filters=200, kernel_size=(3, 3), strides=1, activation='relu', padding='same'))

    # max reduce
    model.add(GlobalMaxPooling2D(data_format='channels_last'))

    # classification
    model.add(Dense(num_classes, activation='sigmoid'))

    print(model.summary())

    return model
