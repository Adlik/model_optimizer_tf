# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tensorflow as tf

"""
https://arxiv.org/pdf/1512.03385.pdf  Deep Residual Learning for Image Recognition
|------------|-------------|----------|----------|----------|-----------|-----------|
| layer name | output size | 18-layer | 34-layer | 50-layer | 101-layer | 152-layer |
|------------|-------------|----------|----------|----------|-----------|-----------|

"""
L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def _gen_l2_regularizer(use_l2_regularizer=True):
  return tf.keras.regularizers.l2(L2_WEIGHT_DECAY) if use_l2_regularizer else None


def resnet(layer_num, num_classes=1001, use_l2_regularizer=True, is_training=True):
    if layer_num == 18:
        stack_block_nums = [2, 2, 2, 2]
    elif layer_num == 34 or layer_num == 50:
        stack_block_nums = [3, 4, 6, 3]
    elif layer_num == 101:
        stack_block_nums = [3, 4, 23, 3]
    elif layer_num == 152:
        stack_block_nums = [3, 8, 36, 3]
    inputs = tf.keras.layers.Input(shape=(224, 224, 3), name='input')
    x = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name='pad1')(inputs)
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='valid', activation=None, use_bias=False,
                               kernel_initializer='he_normal',
                               kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                               name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x, training=is_training)
    x = tf.keras.layers.Activation('relu', name='act1')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    num_stacked_blocks = 4
    num_filters = 64
    for stage in range(num_stacked_blocks):
        for block in range(stack_block_nums[stage]):
            if layer_num < 50:
                x = residual_block(stage, block, x, num_filters, 3, is_training)
            else:
                x = bottleneck_block(stage, block, x, num_filters, 3, is_training)
        num_filters *= 2
    x = tf.keras.layers.AveragePooling2D(pool_size=7, name='avg1')(x)
    x = tf.keras.layers.Flatten(name='flat1')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax',
                                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                    kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                                    bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                                    name='dense1')(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def resnet_18(is_training):
    return resnet(18, is_training=is_training)


def resnet_34(is_training):
    return resnet(34, is_training=is_training)


def resnet_50(is_training):
    return resnet(50, is_training=is_training)


def residual_block(stage, block_num, input_data, filters, kernel_size, is_training):
    if stage != 0 and block_num == 0:
        strides = 2
    else:
        strides = 1
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False,
                               name='res'+str(stage)+'_conv2d_'+str(block_num)+'_1')(input_data)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5,
                                           name='res'+str(stage)+'_bn_'+str(block_num)+'_1')(x, training=is_training)
    x = tf.keras.layers.Activation('relu', name='res'+str(stage)+'_act_'+str(block_num)+'_1')(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, activation=None, padding='same', use_bias=False,
                               name='res'+str(stage)+'_conv2d_'+str(block_num)+'_2')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5,
                                           name='res'+str(stage)+'_bn_'+str(block_num)+'_2')(x, training=is_training)
    if stage != 0 and block_num == 0:
        input_data = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=strides, padding='same', use_bias=False,
                                            name='res'+str(stage)+'_conv2d_'+str(block_num)+'_3')(input_data)
    x = tf.keras.layers.Add(name='res'+str(stage)+'_add_'+str(block_num)+'_1')([x, input_data])
    x = tf.keras.layers.Activation('relu', name='res'+str(stage)+'_act_'+str(block_num)+'_2')(x)
    return x


def bottleneck_block(stage, block_num, input_data, filters, kernel_size, is_training, use_l2_regularizer=True):
    if stage != 0 and block_num == 0:
        strides = 2
    else:
        strides = 1
    x = tf.keras.layers.Conv2D(filters, 1, strides=1, use_bias=False,
                               kernel_initializer='he_normal',
                               kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                               name='res'+str(stage)+'_conv2d_'+str(block_num)+'_1')(input_data)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5,
                                           name='res'+str(stage)+'_bn_'+str(block_num)+'_1')(x, training=is_training)
    x = tf.keras.layers.Activation('relu', name='res'+str(stage)+'_act_'+str(block_num)+'_1')(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
                               activation=None, use_bias=False, kernel_initializer='he_normal',
                               kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                               name='res'+str(stage)+'_conv2d_'+str(block_num)+'_2')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5,
                                           name='res'+str(stage)+'_bn_'+str(block_num)+'_2')(x, training=is_training)
    x = tf.keras.layers.Activation('relu', name='res'+str(stage)+'_act_'+str(block_num)+'_2')(x)
    x = tf.keras.layers.Conv2D(filters*4, 1, strides=1, use_bias=False,
                               kernel_initializer='he_normal',
                               kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                               name='res'+str(stage)+'_conv2d_'+str(block_num)+'_3')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5,
                                           name='res'+str(stage)+'_bn_'+str(block_num)+'_3')(x, training=is_training)
    if block_num == 0:
        input_data = tf.keras.layers.Conv2D(filters*4, kernel_size=1, strides=strides, use_bias=False,
                                            kernel_initializer='he_normal',
                                            kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                                            name='res'+str(stage)+'_conv2d_'+str(block_num)+'_4')(input_data)
        input_data = tf.keras.layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5,
            name='res'+str(stage)+'_bn_'+str(block_num)+'_4')(input_data, training=is_training)
    x = tf.keras.layers.Add(name='res'+str(stage)+'_add_'+str(block_num)+'_1')([x, input_data])
    x = tf.keras.layers.Activation('relu', name='res'+str(stage)+'_act_'+str(block_num))(x)
    return x




