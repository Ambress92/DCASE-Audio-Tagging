#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import Conv2D, BatchNormalization, Activation, Dropout, Lambda, AveragePooling2D, Input, concatenate
import keras.backend as K
from keras.models import Model

def densenet(data_format, num_classes, growth_rate):
    ini_filters = 32

    input = Input(shape=data_format)
    model =Conv2D(ini_filters, (5, 5), strides=2, activation='relu', padding='same',
                     kernel_initializer='he_normal', use_bias=False)(input)
    model =BatchNormalization(momentum=0.9, axis=-1)(model)
    model =AveragePooling2D((2, 2), strides=(2, 2))(model)
    out1 =Dropout(0.3)(model)

    pooled1_input = Lambda(lambda x: K.tile(AveragePooling2D((4,4), strides=(4,4))(x), [1,1,1,growth_rate]))(input)
    input1 = concatenate([pooled1_input, out1])

    model = Conv2D(2 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=False)(input1)
    out2 =BatchNormalization(momentum=0.9, axis=-1)(model)

    out1 = Lambda(lambda x: K.tile(x, [1,1,1,4]))(out1)
    input2 = concatenate([pooled1_input, out1, out2], axis=-1)

    model = Conv2D(2 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=False)(input2)
    model =BatchNormalization(momentum=0.9, axis=-1)(model)
    model =AveragePooling2D((2, 3), strides=(2, 3))(model)
    out3 =Dropout(0.3)(model)
    pooled1_out1 = AveragePooling2D((2,3), strides=(2,3))(out1)
    pooled1_out2 = Lambda(lambda x: K.tile(AveragePooling2D((2,3), strides=(2,3))(x), [1,1,1,2]))(out2)
    pooled2_input = AveragePooling2D((2,3), strides=(2,3))(pooled1_input)


    input3 = concatenate([pooled2_input, pooled1_out1, pooled1_out2, out3], axis=-1)

    model = Conv2D(4 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=False)(input3)
    out3 = Lambda(lambda x: K.tile(x, [1,1,1,2]))(out3)
    out4 =BatchNormalization(momentum=0.9, axis=-1)(model)


    input4 = concatenate([pooled2_input, pooled1_out1, pooled1_out2, out3, out4], axis=-1)

    model = Conv2D(4 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=False)(input4)
    model =BatchNormalization(momentum=0.9, axis=-1)(model)
    model =AveragePooling2D((2, 3), strides=(2, 3))(model)
    out5 =Dropout(0.3)(model)

    pooled3_input = AveragePooling2D((2,3), strides=(2,3))(pooled2_input)
    pooled2_out1 = AveragePooling2D((2,3), strides=(2,3))(pooled1_out1)
    pooled2_out2 = AveragePooling2D((2,3), strides=(2,3))(pooled1_out2)
    pooled1_out3 = AveragePooling2D((2,3), strides=(2,3))(out3)
    pooled1_out4 = AveragePooling2D((2,3), strides=(2,3))(out4)

    input5 = concatenate([pooled3_input, pooled2_out1, pooled2_out2, pooled1_out3, pooled1_out4, out5], axis=-1)

    model = Conv2D(4 * ini_filters, (1, 1), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False)(input5)
    out6 = BatchNormalization(momentum=0.9, axis=-1)(model)

    input6 = concatenate([pooled3_input, pooled2_out1, pooled2_out2, pooled1_out3, pooled1_out4, out5, out6], axis=-1)

    model = Conv2D(4 * ini_filters, (1, 1), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False)(input6)
    model =BatchNormalization(momentum=0.9, axis=-1)(model)
    out7 =Dropout(0.5)(model)

    input7 = concatenate([pooled3_input, pooled2_out1, pooled2_out2, pooled1_out3, pooled1_out4, out5, out6, out7], axis=-1)

    # classification block
    model =Conv2D(num_classes, (1, 1), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=False)(input7)
    model =Lambda(lambda x: K.mean(x, axis=1))(model)
    model =Lambda(lambda x: K.max(x, axis=1))(model)
    # model =GlobalAveragePooling2D(data_format='channels_last'))
    final_out = Activation(activation='softmax')(model)

    model = Model(inputs=[input], outputs=[final_out])

    print(model.summary())

    return model


def architecture(data_format, num_classes, growth_rate):
    """
    Instantiates a network model for a given dictionary of input/output
    tensor formats (dtype, shape) and a given configuration.
    """
    return densenet(data_format, num_classes, growth_rate)


