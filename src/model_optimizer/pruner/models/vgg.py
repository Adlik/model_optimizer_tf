# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VGG models
"""
import tensorflow as tf

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def vgg_16(is_training, name, num_classes=1001, use_l2_regularizer=True):
    """
    VGG-16 model
    :param is_training: if training or not
    :param name: the model name
    :param num_classes: classification class
    :param use_l2_regularizer: if use l2 regularizer or not
    :return:
    """
    return vgg(ver='D', is_training=is_training, name=name, num_classes=num_classes,
               use_l2_regularizer=use_l2_regularizer)


def vgg_19(is_training, name, num_classes=1001, use_l2_regularizer=True):
    """
    VGG-19 model
    :param is_training: if training or not
    :param name: the model name
    :param num_classes: classification class
    :param use_l2_regularizer: if use l2 regularizer or not
    :return:
    """
    return vgg(ver='E', is_training=is_training, name=name, num_classes=num_classes,
               use_l2_regularizer=use_l2_regularizer)


def vgg_m_16(is_training, name, num_classes=10, use_l2_regularizer=True):
    """
    VGG-M-16 model
    :param is_training: if training or not
    :param name: the model name
    :param num_classes: classification class
    :param use_l2_regularizer: if use l2 regularizer or not
    :return:
    """
    return vgg_m(ver='D', is_training=is_training, name=name, num_classes=num_classes,
                 use_l2_regularizer=use_l2_regularizer)


def vgg_m_19(is_training, name, num_classes=10, use_l2_regularizer=True):
    """
    VGG-M-19 model
    :param is_training: if training or not
    :param name: the model name
    :param num_classes: classification class
    :param use_l2_regularizer: if use l2 regularizer or not
    :return:
    """
    return vgg_m(ver='E', is_training=is_training, name=name, num_classes=num_classes,
                 use_l2_regularizer=use_l2_regularizer)


def _gen_l2_regularizer(use_l2_regularizer=True):
    return tf.keras.regularizers.l2(L2_WEIGHT_DECAY) if use_l2_regularizer else None


def _vgg_blocks(block, conv_num, filters, x, is_training, use_l2_regularizer=True):
    for i in range(1, conv_num+1):
        x = tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same', activation=None, use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                                   name='block'+str(block)+'_conv'+str(i))(x)
        x = tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,
                                               name='block'+str(block)+'_bn'+str(i))(x, training=is_training)
        x = tf.keras.layers.Activation('relu', name='block'+str(block)+'_act'+str(i))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same',
                                  name='maxpool2d' + str(block))(x)
    return x


def vgg(ver, is_training, name, num_classes=1001, use_l2_regularizer=True):
    """
    VGG models
    :param ver: 'D' or 'E'
    :param is_training: if training or not
    :param name: the model name
    :param num_classes: classification class
    :param use_l2_regularizer: if use l2 regularizer or not
    :return:
    """
    if ver == 'D':
        block_conv_nums = [2, 2, 3, 3, 3]
        conv_filter_nums = [64, 128, 256, 512, 512]
    elif ver == 'E':
        block_conv_nums = [2, 2, 4, 4, 4]
        conv_filter_nums = [64, 128, 256, 512, 512]
    inputs = tf.keras.layers.Input(shape=(224, 224, 3), name='input')
    x = None
    for i, conv_num in enumerate(block_conv_nums):
        filters = conv_filter_nums[i]
        block = i + 1
        if x is None:
            x = inputs
        x = _vgg_blocks(block, conv_num, filters, x, is_training, use_l2_regularizer=True)

    x = tf.keras.layers.Flatten(name='flat1')(x)
    x = tf.keras.layers.Dense(4096, activation='relu',
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                              kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                              bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                              name='fc1')(x)
    x = tf.keras.layers.Dense(4096, activation='relu',
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                              kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                              bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                              name='fc2')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax',
                                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                    kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                                    bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                                    name='fc3')(x)
    model = tf.keras.Model(inputs, outputs, name=name)
    return model


def vgg_m(ver, is_training, name, num_classes=10, use_l2_regularizer=True):
    """
    VGG-M models
    :param ver: 'D' or 'E'
    :param is_training: if training or not
    :param name: the model name
    :param num_classes: classification class
    :param use_l2_regularizer: if use l2 regularizer or not
    :return:
    """
    if ver == 'D':
        block_conv_nums = [2, 2, 3, 3, 3]
        conv_filter_nums = [64, 128, 256, 512, 512]
    elif ver == 'E':
        block_conv_nums = [2, 2, 4, 4, 4]
        conv_filter_nums = [64, 128, 256, 512, 512]
    inputs = tf.keras.layers.Input(shape=(32, 32, 3), name='input')
    x = None
    for i, conv_num in enumerate(block_conv_nums):
        filters = conv_filter_nums[i]
        block = i + 1
        if x is None:
            x = inputs
        x = _vgg_blocks(block, conv_num, filters, x, is_training, use_l2_regularizer=True)

    x = tf.keras.layers.Flatten(name='flat1')(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax',
                                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                    kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                                    bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                                    name='fc2')(x)
    model = tf.keras.Model(inputs, outputs, name=name)
    return model
