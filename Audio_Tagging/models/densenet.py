#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Lambda, AveragePooling2D, Input
import keras.backend as K

def densenet(data_format, num_classes, growth_rate):
    ini_filters = 64

    input = Input(shape=data_format)
    model =Conv2D(ini_filters, (5, 5), strides=2, activation='relu', padding='same',
                     kernel_initializer='he_normal', use_bias=False)(input)
    model =BatchNormalization(momentum=0.9, axis=-1)(model)
    model =AveragePooling2D((2, 2), strides=(2, 2))(model)
    out1 =Dropout(0.3)(model)

    pooled1_input = K.tile(AveragePooling2D((4,4), strides=(4,4))(input), [1,1,1,growth_rate])
    input1 = K.concatenate([pooled1_input, out1])

    model = Conv2D(2 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=False)(input1)
    out2 =BatchNormalization(momentum=0.9, axis=-1)(model)

    input2 = K.concatenate([pooled1_input, K.tile(out1, [1,1,1,2]), out2], axis=-1)

    model = Conv2D(2 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=False)(input2)
    model =BatchNormalization(momentum=0.9, axis=-1)(model)
    model =AveragePooling2D((2, 2), strides=(2, 2))(model)
    out3 =Dropout(0.3)(model)
    pooled1_out1 = AveragePooling2D((2,2), strides=(2,2))(out1)
    pooled1_out2 = AveragePooling2D((2,2), strides=(2,2))(out2)
    pooled2_input = AveragePooling2D((2,2), strides=(2,2))(pooled1_input)

    input3 = K.concatenate([pooled2_input, K.tile(pooled1_out1, [1,1,1,2]), pooled1_out2, out3], axis=-1)

    model = Conv2D(4 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=False)(input3)
    out4 =BatchNormalization(momentum=0.9, axis=-1)(model)

    input4 = K.concatenate([pooled2_input, K.tile(pooled1_out1, [1,1,1,2]), pooled1_out2, out3, out4], axis=-1)

    model = Conv2D(4 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=False)(input4)
    model =BatchNormalization(momentum=0.9, axis=-1)(model)
    model =AveragePooling2D((2, 3), strides=(2, 3))(model)
    out5 =Dropout(0.3)(model)

    pooled3_input = AveragePooling2D((2,3), strides=(2,3))(pooled2_input)
    pooled2_out1 = AveragePooling2D((2,3), strides=(2,3))(pooled1_out1)
    pooled2_out2 = AveragePooling2D((2,3), strides=(2,3))(pooled1_out2)
    pooled1_out3 = AveragePooling2D((2,3), strides=(2,3))(out3)
    pooled1_out4 = AveragePooling2D((2,3), strides=(2,3))(out4)

    input5 = K.concatenate([pooled3_input, K.tile(pooled2_out1, [1,1,1,2]), pooled2_out2, pooled1_out3, pooled1_out4, out5], axis=-1)

    model = Conv2D(8 * ini_filters, (1, 1), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False)(input5)
    out6 = BatchNormalization(momentum=0.9, axis=-1)(model)

    input6 = K.concatenate([pooled3_input, K.tile(pooled2_out1, [1,1,1,2]), pooled2_out2, pooled1_out3, pooled1_out4, out5, out6], axis=-1)

    model = Conv2D(8 * ini_filters, (1, 1), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False)(input6)
    model =BatchNormalization(momentum=0.9, axis=-1)(model)
    out7 =Dropout(0.5)(model)

    input7 = K.concatenate([pooled3_input, K.tile(pooled2_out1, [1,1,1,2]), pooled2_out2, pooled1_out3, pooled1_out4, out5, out6, out7], axis=-1)

    # classification block
    model =Conv2D(num_classes, (1, 1), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=False)(input7)
    model =Lambda(lambda x: K.mean(x, axis=1))(model)
    model =Lambda(lambda x: K.max(x, axis=1))(model)
    # model =GlobalAveragePooling2D(data_format='channels_last'))
    model =Activation(activation='softmax')(model)

    print(model.summary())

    return model


def architecture(data_format, num_classes, growth_rate):
    """
    Instantiates a network model for a given dictionary of input/output
    tensor formats (dtype, shape) and a given configuration.
    """
    return densenet(data_format, num_classes, growth_rate)


