# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Lenet model
"""
import tensorflow as tf


def lenet(name, is_training=True):
    """
    This implements a slightly modified LeNet-5 [LeCun et al., 1998a]
    :param name: the model name
    :param is_training: if training or not
    :return: LeNet model
    """
    input_ = tf.keras.layers.Input(shape=(28, 28, 1), name='input')
    x = tf.keras.layers.Conv2D(filters=6,
                               kernel_size=(3, 3),
                               padding='same',
                               activation=None,
                               name='conv2d_1')(input_)
    x = tf.keras.layers.BatchNormalization(name='bn_1')(x, training=is_training)
    x = tf.keras.layers.Activation('relu', name='act_1')(x)
    x = tf.keras.layers.AveragePooling2D(name='pool_1')(x)
    x = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(3, 3),
                               activation=None,
                               name='conv2d_2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_2')(x, training=is_training)
    x = tf.keras.layers.Activation('relu', name='act_2')(x)
    x = tf.keras.layers.AveragePooling2D(name='pool_2')(x)
    x = tf.keras.layers.Flatten(name='flatten')(x)
    x = tf.keras.layers.Dense(120, activation='relu', name='dense_1')(x)
    x = tf.keras.layers.Dense(84, activation='relu', name='dense_2')(x)
    output_ = tf.keras.layers.Dense(10, activation='softmax', name='dense_3')(x)
    model = tf.keras.Model(input_, output_, name=name)
    return model
