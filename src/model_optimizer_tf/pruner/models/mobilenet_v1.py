# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
   https://arxiv.org/abs/1704.04861  MobileNets: Efficient Convolutional Neural Networks for
   Mobile Vision Applications
   Adapted from tf.keras.applications.mobilenet.MobileNetV2().
"""
import tensorflow as tf
from .config import ModelConfig

_config = ModelConfig(l2_weight_decay=0.00004, batch_norm_decay=0.95, batch_norm_epsilon=0.001, std_dev=0.09)
L2_WEIGHT_DECAY = _config.l2_weight_decay
BATCH_NORM_DECAY = _config.batch_norm_decay
BATCH_NORM_EPSILON = _config.batch_norm_epsilon
STD_DEV = _config.std_dev


def _gen_l2_regularizer(use_l2_regularizer=True):
    return tf.keras.regularizers.l2(L2_WEIGHT_DECAY) if use_l2_regularizer else None


def _gen_initializer(use_initializer=True):
    return tf.keras.initializers.TruncatedNormal(stddev=STD_DEV) if use_initializer else None


def mobilenet_v1_0_25(num_classes=1000,
                      dropout_prob=1e-3,
                      is_training=True,
                      depth_multiplier=1,
                      classifier_activation='softmax'):
    """
    Build mobilenet_v1_0.25 model
    :param num_classes:
    :param dropout_prob:
    :param is_training:
    :param depth_multiplier:
    :param classifier_activation: classifier_activation can only be None or "softmax"
    :return:
    """
    return _mobilenet_v1(num_classes, dropout_prob, is_training, scale=0.25,
                         depth_multiplier=depth_multiplier, classifier_activation=classifier_activation)


def mobilenet_v1_0_5(num_classes=1000,
                     dropout_prob=1e-3,
                     is_training=True,
                     depth_multiplier=1,
                     classifier_activation='softmax'):
    """
    Build mobilenet_v1_0.5 model
    :param num_classes:
    :param dropout_prob:
    :param is_training:
    :param depth_multiplier:
    :param classifier_activation: classifier_activation can only be None or "softmax"
    :return:
    """
    return _mobilenet_v1(num_classes, dropout_prob, is_training, scale=0.5,
                         depth_multiplier=depth_multiplier, classifier_activation=classifier_activation)


def mobilenet_v1_0_75(num_classes=1000,
                      dropout_prob=1e-3,
                      is_training=True,
                      depth_multiplier=1,
                      classifier_activation='softmax'):
    """
    Build mobilenet_v1_0.75 model
    :param num_classes:
    :param dropout_prob:
    :param is_training:
    :param depth_multiplier:
    :param classifier_activation: classifier_activation can only be None or "softmax"
    :return:
    """
    return _mobilenet_v1(num_classes, dropout_prob, is_training, scale=0.75,
                         depth_multiplier=depth_multiplier, classifier_activation=classifier_activation)


def mobilenet_v1_1(name, num_classes=1000,
                   dropout_prob=1e-3,
                   is_training=True,
                   depth_multiplier=1,
                   classifier_activation='softmax'):
    """
    Build mobilenet_v1_1.0 model
    :param name: the model name
    :param num_classes:
    :param dropout_prob:
    :param is_training:
    :param depth_multiplier:
    :param classifier_activation: classifier_activation can only be None or "softmax"
    :return:
    """
    return _mobilenet_v1(name, num_classes, dropout_prob, is_training, scale=1.0,
                         depth_multiplier=depth_multiplier, classifier_activation=classifier_activation)


def _mobilenet_v1(name, num_classes=1000,
                  dropout_prob=1e-3,
                  is_training=True,
                  scale=1.0,
                  depth_multiplier=1,
                  classifier_activation='softmax'):
    """
    Build mobilenet_v1 model
    :param name: the model name
    :param num_classes:
    :param dropout_prob:
    :param is_training:
    :param scale:
    :param depth_multiplier:
    :param classifier_activation: classifier_activation can only be None or "softmax"
    :return:
    """
    inputs = tf.keras.layers.Input(shape=(224, 224, 3), name='input')
    x = _conv_block(inputs, int(32*scale), is_training=is_training, strides=(2, 2))
    x = _depthwise_conv_block(x, int(64*scale), is_training, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(x, int(128*scale), is_training, depth_multiplier,
                              strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, int(128*scale), is_training, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(x, int(256*scale), is_training, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, int(256*scale), is_training, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x, int(512*scale), is_training, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, int(512*scale), is_training, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, int(512*scale), is_training, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, int(512*scale), is_training, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, int(512*scale), is_training, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, int(512*scale), is_training, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(x, int(1024*scale), is_training, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, int(1024*scale), is_training, depth_multiplier, block_id=13)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Reshape(target_shape=(1, 1, int(1024 * scale)), name='reshape_1')(x)
    if is_training:
        x = tf.keras.layers.Dropout(dropout_prob, name='dropout')(x)
    x = tf.keras.layers.Conv2D(num_classes, (1, 1),
                               padding='same',
                               kernel_initializer=_gen_initializer(),
                               kernel_regularizer=_gen_l2_regularizer(),
                               name='conv_preds')(x)
    x = tf.keras.layers.Reshape((num_classes,), name='reshape_2')(x)
    if classifier_activation == 'softmax':
        outputs = tf.keras.layers.Activation('softmax', name='act_softmax')(x)
    else:
        outputs = x
    model = tf.keras.Model(inputs, outputs, name=name)
    return model


def _conv_block(inputs, filters, is_training=True, kernel=(3, 3), strides=(1, 1)):
    x = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(inputs)
    x = tf.keras.layers.Conv2D(filters, kernel,
                               padding='valid',
                               use_bias=False,
                               strides=strides,
                               kernel_initializer=_gen_initializer(),
                               kernel_regularizer=_gen_l2_regularizer(),
                               name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY,
                                           epsilon=BATCH_NORM_EPSILON, name='conv1_bn')(x, training=is_training)
    return tf.keras.layers.ReLU(6., name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, is_training=True,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    if strides == (1, 1):
        x = inputs
    else:
        x = tf.keras.layers.ZeroPadding2D(((0, 1), (0, 1)),
                                          name=f'conv_pad_{block_id}')(inputs)
    x = tf.keras.layers.DepthwiseConv2D((3, 3),
                                        padding='same' if strides == (1, 1) else 'valid',
                                        depth_multiplier=depth_multiplier,
                                        strides=strides,
                                        use_bias=False,
                                        kernel_initializer=_gen_initializer(),
                                        kernel_regularizer=_gen_l2_regularizer(False),
                                        name=f'conv_dw_{block_id}')(x)
    x = tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY,
                                           epsilon=BATCH_NORM_EPSILON,
                                           name=f'conv_dw_{block_id}_bn')(x, training=is_training)
    x = tf.keras.layers.ReLU(6., name=f'conv_dw_{block_id}_relu')(x)

    x = tf.keras.layers.Conv2D(pointwise_conv_filters, (1, 1),
                               padding='same',
                               use_bias=False,
                               strides=(1, 1),
                               kernel_initializer=_gen_initializer(),
                               kernel_regularizer=_gen_l2_regularizer(),
                               name=f'conv_pw_{block_id}')(x)
    x = tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY,
                                           epsilon=BATCH_NORM_EPSILON,
                                           name=f'conv_pw_{block_id}_bn')(x, training=is_training)
    return tf.keras.layers.ReLU(6., name=f'conv_pw_{block_id}_relu')(x)
