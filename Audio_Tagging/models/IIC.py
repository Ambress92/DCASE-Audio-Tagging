#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Lambda, Input
import keras.backend as K
from keras import Model
import tensorflow as tf
import sys

def semi_supervised_IIC(input_shape, num_classes, final_act):
    ini_filters = 64
    input = Input(input_shape)
    augmented_input = Input(input_shape)

    conv1_1 = Conv2D(ini_filters, (5, 5), strides=2, activation='relu', padding='same',
                     kernel_initializer='he_normal', use_bias=False)
    batch_norm1_1 = BatchNormalization(momentum=0.9, axis=-1)
    conv1_2 = Conv2D(ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
                     use_bias=False)
    batch_norm1_2 = BatchNormalization(momentum=0.9, axis=-1)
    max_pool1 = MaxPooling2D((2, 2), strides=(2, 2))
    dropout1 = Dropout(0.3)

    conv2_1 = \
        Conv2D(2 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False)
    batch_norm2_1 = BatchNormalization(momentum=0.9, axis=-1)
    conv2_2 = \
        Conv2D(2 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False)
    batch_norm2_2 = BatchNormalization(momentum=0.9, axis=-1)
    max_pool2 = MaxPooling2D((2, 2), strides=(2, 2))
    dropout2 = Dropout(0.3)

    conv3_1 = \
        Conv2D(4 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False)
    batch_norm3_1 = BatchNormalization(momentum=0.9, axis=-1)
    dropout3_1 = Dropout(0.3)
    conv3_2 = \
        Conv2D(4 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False)
    batch_norm3_2 = BatchNormalization(momentum=0.9, axis=-1)
    dropout3_2 = Dropout(0.3)
    conv3_3 = \
        Conv2D(6 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False)
    batch_norm3_3 = BatchNormalization(momentum=0.9, axis=-1)
    dropout3_3 = Dropout(0.3)
    conv3_4 = \
        Conv2D(6 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False)
    batch_norm3_4 = BatchNormalization(momentum=0.9, axis=-1)
    max_pool3 = MaxPooling2D((2, 2), strides=(2, 2))
    dropout3_4 = Dropout(0.3)

    conv4_1 = \
        Conv2D(8 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False)
    batch_norm4_1 = BatchNormalization(momentum=0.9, axis=-1)
    conv4_2 = \
        Conv2D(8 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False)
    batch_norm4_2 = BatchNormalization(momentum=0.9, axis=-1)
    max_pool4 = MaxPooling2D((1, 2), strides=(1, 2))
    dropout4 = Dropout(0.3)

    conv5_1 = \
        Conv2D(8 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False)
    batch_norm5_1 = BatchNormalization(momentum=0.9, axis=-1)
    conv5_2 = \
        Conv2D(8 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False)
    batch_norm5_2 = BatchNormalization(momentum=0.9, axis=-1)
    max_pool5 = MaxPooling2D((1, 2), strides=(1, 2))
    dropout5 = Dropout(0.3)

    conv6_1 = \
        Conv2D(8 * ini_filters, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False)
    batch_norm6_1 = BatchNormalization(momentum=0.9, axis=-1)
    dropout6_1 = Dropout(0.5)
    conv6_2 = \
        Conv2D(8 * ini_filters, (1, 1), strides=1, activation='relu', padding='same', kernel_initializer='he_normal',
               use_bias=False)
    batch_norm6_2 = BatchNormalization(momentum=0.9, axis=-1)
    dropout6_2 = Dropout(0.5)

    # clustering head
    conv7 = Conv2D(num_classes, (1, 1), strides=1, activation='relu', padding='same', kernel_initializer='he_normal')
    lambda_7_1 = Lambda(lambda x: K.mean(x, axis=1))
    lambda_7_2 = Lambda(lambda x: K.max(x, axis=1))
    cluster_output1 = Activation(activation="softmax", name='cluster_output')
    cluster_output2 = Activation(activation="softmax", name='cluster_output_aug')

    # classification block
    conv7_clf = Conv2D(num_classes, (1, 1), strides=1, activation='linear', padding='same', kernel_initializer='he_normal')
    lambda_7_1_clf = Lambda(lambda x: K.mean(x, axis=1))
    lambda_7_2_clf = Lambda(lambda x: K.max(x, axis=1))
    output_clf = Activation(activation=final_act, name='clf_output')
    output_clf_augmented = Activation(activation=final_act, name='clf_output_augmented')

    model = conv1_1(input)
    model = batch_norm1_1(model)
    model = conv1_2(model)
    model = batch_norm1_2(model)
    model = max_pool1(model)
    model = dropout1(model)
    model = conv2_1(model)
    model = batch_norm2_1(model)
    model = conv2_2(model)
    model = batch_norm2_2(model)
    model = max_pool2(model)
    model = dropout2(model)
    model = conv3_1(model)
    model = batch_norm3_1(model)
    model = dropout3_1(model)
    model = conv3_2(model)
    model = batch_norm3_2(model)
    model = dropout3_2(model)
    model = conv3_3(model)
    model = batch_norm3_3(model)
    model = dropout3_3(model)
    model = conv3_4(model)
    model = batch_norm3_4(model)
    model = max_pool3(model)
    model = dropout3_4(model)
    model = conv4_1(model)
    model = batch_norm4_1(model)
    model = conv4_2(model)
    model = batch_norm4_2(model)
    model = max_pool4(model)
    model = dropout4(model)
    model = conv5_1(model)
    model = batch_norm5_1(model)
    model = conv5_2(model)
    model = batch_norm5_2(model)
    model = max_pool5(model)
    model = dropout5(model)
    model = conv6_1(model)
    model = batch_norm6_1(model)
    model = dropout6_1(model)
    model = conv6_2(model)
    model = batch_norm6_2(model)
    clf_fork = dropout6_2(model)
    model = conv7(clf_fork)
    model = lambda_7_1(model)
    model = lambda_7_2(model)
    output1 = cluster_output1(model)

    augmented_model = conv1_1(augmented_input)
    augmented_model = batch_norm1_1(augmented_model)
    augmented_model = conv1_2(augmented_model)
    augmented_model = batch_norm1_2(augmented_model)
    augmented_model = max_pool1(augmented_model)
    augmented_model = dropout1(augmented_model)
    augmented_model = conv2_1(augmented_model)
    augmented_model = batch_norm2_1(augmented_model)
    augmented_model = conv2_2(augmented_model)
    augmented_model = batch_norm2_2(augmented_model)
    augmented_model = max_pool2(augmented_model)
    augmented_model = dropout2(augmented_model)
    augmented_model = conv3_1(augmented_model)
    augmented_model = batch_norm3_1(augmented_model)
    augmented_model = dropout3_1(augmented_model)
    augmented_model = conv3_2(augmented_model)
    augmented_model = batch_norm3_2(augmented_model)
    augmented_model = dropout3_2(augmented_model)
    augmented_model = conv3_3(augmented_model)
    augmented_model = batch_norm3_3(augmented_model)
    augmented_model = dropout3_3(augmented_model)
    augmented_model = conv3_4(augmented_model)
    augmented_model = batch_norm3_4(augmented_model)
    augmented_model = max_pool3(augmented_model)
    augmented_model = dropout3_4(augmented_model)
    augmented_model = conv4_1(augmented_model)
    augmented_model = batch_norm4_1(augmented_model)
    augmented_model = conv4_2(augmented_model)
    augmented_model = batch_norm4_2(augmented_model)
    augmented_model = max_pool4(augmented_model)
    augmented_model = dropout4(augmented_model)
    augmented_model = conv5_1(augmented_model)
    augmented_model = batch_norm5_1(augmented_model)
    augmented_model = conv5_2(augmented_model)
    augmented_model = batch_norm5_2(augmented_model)
    augmented_model = max_pool5(augmented_model)
    augmented_model = dropout5(augmented_model)
    augmented_model = conv6_1(augmented_model)
    augmented_model = batch_norm6_1(augmented_model)
    augmented_model = dropout6_1(augmented_model)
    augmented_model = conv6_2(augmented_model)
    augmented_model = batch_norm6_2(augmented_model)
    augmented_model = dropout6_2(augmented_model)
    augmented_model = conv7(augmented_model)
    augmented_model = lambda_7_1(augmented_model)
    augmented_model = lambda_7_2(augmented_model)
    output2 = cluster_output2(augmented_model)

    # classification block for normal spectrogram
    clf_model = conv7_clf(clf_fork)
    clf_model = lambda_7_1_clf(clf_model)
    clf_model = lambda_7_2_clf(clf_model)
    clf_output = output_clf(clf_model)

    # classification block for augmented image
    clf_model_augmented = conv7_clf(clf_fork)
    clf_model_augmented = lambda_7_1_clf(clf_model_augmented)
    clf_model_augmented = lambda_7_2_clf(clf_model_augmented)
    clf_output_augmented = output_clf_augmented(clf_model_augmented)

    model = Model(inputs=[input, augmented_input], outputs=[output1, output2, clf_output, clf_output_augmented])

    P = (tf.expand_dims(output1, axis=2) * tf.expand_dims(output2, axis=1))
    P = tf.reduce_sum(P, axis=0)
    P = ((P + tf.transpose(P)) / 2) / tf.reduce_sum(P)
    EPS = tf.fill(P.shape, sys.float_info.epsilon)
    P = tf.where(P < EPS, EPS, P)
    Pi = tf.reshape(tf.reduce_sum(P, axis=1), (num_classes, 1))
    Pj = tf.reshape(tf.reduce_sum(P, axis=0), (1, num_classes))
    loss = tf.reduce_sum(P * (tf.log(Pi) + tf.log(Pj) - tf.log(P)))

    model.add_loss(loss)

    print(model.summary())
    return model

def architecture(data_format, num_classes):
    """
    Instantiates a network model for a given dictionary of input/output
    tensor formats (dtype, shape) and a given configuration.
    """
    return semi_supervised_IIC(data_format, num_classes, final_act='softmax')


