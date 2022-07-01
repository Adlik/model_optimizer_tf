# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
https://arxiv.org/abs/1801.04381  MobileNetV2: Inverted Residuals and Linear Bottlenecks
   Adapted from tf.keras.applications.mobilenet.MobileNetV2().
"""
import tensorflow as tf
from .config import ModelConfig

_config = ModelConfig(l2_weight_decay=0.00004, batch_norm_decay=0.99, batch_norm_epsilon=0.001, std_dev=0.09)
L2_WEIGHT_DECAY = _config.l2_weight_decay
BATCH_NORM_DECAY = _config.batch_norm_decay
BATCH_NORM_EPSILON = _config.batch_norm_epsilon
STD_DEV = _config.std_dev


def _gen_l2_regularizer(use_l2_regularizer=True):
    return tf.keras.regularizers.l2(L2_WEIGHT_DECAY) if use_l2_regularizer else None


def _gen_initializer(use_initializer=True):
    return tf.keras.initializers.TruncatedNormal(stddev=STD_DEV) if use_initializer else None


def mobilenet_v2_0_35(num_classes=1000,
                      dropout_prob=1e-3,
                      is_training=True,
                      classifier_activation='softmax'):
    """
    Build mobilenet_v2_0.35 model
    :param num_classes:
    :param dropout_prob:
    :param is_training:
    :param classifier_activation: classifier_activation can only be None or "softmax"
    :return:
    """
    return _mobilenet_v2(num_classes, dropout_prob, is_training,
                         scale=0.35, classifier_activation=classifier_activation)


def mobilenet_v2_0_5(num_classes=1000,
                     dropout_prob=1e-3,
                     is_training=True,
                     classifier_activation='softmax'):
    """
    Build mobilenet_v2_0.5 model
    :param num_classes:
    :param dropout_prob:
    :param is_training:
    :param classifier_activation: classifier_activation can only be None or "softmax"
    :return:
    """
    return _mobilenet_v2(num_classes, dropout_prob, is_training,
                         scale=0.5, classifier_activation=classifier_activation)


def mobilenet_v2_0_75(num_classes=1000,
                      dropout_prob=1e-3,
                      is_training=True,
                      classifier_activation='softmax'):
    """
    Build mobilenet_v2_0.75 model
    :param num_classes:
    :param dropout_prob:
    :param is_training:
    :param classifier_activation: classifier_activation can only be None or "softmax"
    :return:
    """
    return _mobilenet_v2(num_classes, dropout_prob, is_training,
                         scale=0.75, classifier_activation=classifier_activation)


def mobilenet_v2_1(name, num_classes=1000,
                   dropout_prob=1e-3,
                   is_training=True,
                   classifier_activation='softmax'):
    """
    Build mobilenet_v2_1.0 model
    :param name: the model name
    :param num_classes:
    :param dropout_prob:
    :param is_training:
    :param classifier_activation: classifier_activation can only be None or "softmax"
    :return:
    """
    return _mobilenet_v2(name, num_classes, dropout_prob, is_training,
                         scale=1.0, classifier_activation=classifier_activation)


def mobilenet_v2_1_3(name, num_classes=1000,
                     dropout_prob=1e-3,
                     is_training=True,
                     classifier_activation='softmax'):
    """
    Build mobilenet_v2_1.3 model
    :param name: the model name
    :param num_classes:
    :param dropout_prob:
    :param is_training:
    :param classifier_activation: classifier_activation can only be None or "softmax"
    :return:
    """
    return _mobilenet_v2(name, num_classes, dropout_prob, is_training,
                         scale=1.3, classifier_activation=classifier_activation)


def mobilenet_v2_1_4(num_classes=1000,
                     dropout_prob=1e-3,
                     is_training=True,
                     classifier_activation='softmax'):
    """
    Build mobilenet_v2_1.4 model
    :param num_classes:
    :param dropout_prob:
    :param is_training:
    :param classifier_activation: classifier_activation can only be None or "softmax"
    :return:
    """
    return _mobilenet_v2(num_classes, dropout_prob, is_training,
                         scale=1.4, classifier_activation=classifier_activation)


def _mobilenet_v2(name, num_classes=1000,
                  dropout_prob=1e-3,
                  is_training=True,
                  scale=1.0,
                  classifier_activation='softmax'):
    """
    Build mobilenet_v2 model
    :param name: the model name
    :param num_classes:
    :param dropout_prob:
    :param is_training:
    :param scale:
    :param classifier_activation: classifier_activation can only be None or "softmax"
    :return:
    """
    first_block_filters = _make_divisible(32 * scale, 8)
    inputs = tf.keras.layers.Input(shape=(224, 224, 3), name='input')
    x = tf.keras.layers.ZeroPadding2D(
        padding=((0, 1), (0, 1)),
        name='Conv1_pad')(inputs)
    x = tf.keras.layers.Conv2D(
        first_block_filters,
        kernel_size=3,
        strides=(2, 2),
        padding='valid',
        use_bias=False,
        kernel_initializer=_gen_initializer(),
        kernel_regularizer=_gen_l2_regularizer(),
        name='Conv1')(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=BATCH_NORM_EPSILON,
        momentum=BATCH_NORM_DECAY,
        name='bn_Conv1')(x, training=is_training)
    x = tf.keras.layers.ReLU(6., name='Conv1_relu')(x)

    x = _inverted_res_block(
        x, filters=16, scale=scale, stride=1, expansion=1, block_id=0, is_training=is_training)

    x = _inverted_res_block(
        x, filters=24, scale=scale, stride=2, expansion=6, block_id=1, is_training=is_training)
    x = _inverted_res_block(
        x, filters=24, scale=scale, stride=1, expansion=6, block_id=2, is_training=is_training)

    x = _inverted_res_block(
        x, filters=32, scale=scale, stride=2, expansion=6, block_id=3, is_training=is_training)
    x = _inverted_res_block(
        x, filters=32, scale=scale, stride=1, expansion=6, block_id=4, is_training=is_training)
    x = _inverted_res_block(
        x, filters=32, scale=scale, stride=1, expansion=6, block_id=5, is_training=is_training)

    x = _inverted_res_block(
        x, filters=64, scale=scale, stride=2, expansion=6, block_id=6, is_training=is_training)
    x = _inverted_res_block(
        x, filters=64, scale=scale, stride=1, expansion=6, block_id=7, is_training=is_training)
    x = _inverted_res_block(
        x, filters=64, scale=scale, stride=1, expansion=6, block_id=8, is_training=is_training)
    x = _inverted_res_block(
        x, filters=64, scale=scale, stride=1, expansion=6, block_id=9, is_training=is_training)

    x = _inverted_res_block(
        x, filters=96, scale=scale, stride=1, expansion=6, block_id=10, is_training=is_training)
    x = _inverted_res_block(
        x, filters=96, scale=scale, stride=1, expansion=6, block_id=11, is_training=is_training)
    x = _inverted_res_block(
        x, filters=96, scale=scale, stride=1, expansion=6, block_id=12, is_training=is_training)

    x = _inverted_res_block(
        x, filters=160, scale=scale, stride=2, expansion=6, block_id=13, is_training=is_training)
    x = _inverted_res_block(
        x, filters=160, scale=scale, stride=1, expansion=6, block_id=14, is_training=is_training)
    x = _inverted_res_block(
        x, filters=160, scale=scale, stride=1, expansion=6, block_id=15, is_training=is_training)

    x = _inverted_res_block(
        x, filters=320, scale=scale, stride=1, expansion=6, block_id=16, is_training=is_training)
    if scale > 1.0:
        last_block_filters = _make_divisible(1280 * scale, 8)
    else:
        last_block_filters = 1280
    x = tf.keras.layers.Conv2D(last_block_filters, kernel_size=1, use_bias=False,
                               kernel_initializer=_gen_initializer(), kernel_regularizer=_gen_l2_regularizer(),
                               name='Conv_1')(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_DECAY, name='Conv_1_bn')(x, training=is_training)
    x = tf.keras.layers.ReLU(6., name='out_relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Reshape(target_shape=(1, 1, last_block_filters), name='reshape_1')(x)
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


def _inverted_res_block(inputs, filters, scale, stride, expansion, block_id, is_training=True):
    """Inverted ResNet block."""
    in_channels = inputs.shape[-1]
    pointwise_conv_filters = int(filters * scale)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = f'block_{block_id}_'

    if block_id:
        # Expand
        x = tf.keras.layers.Conv2D(
            expansion * in_channels,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=None,
            kernel_initializer=_gen_initializer(),
            kernel_regularizer=_gen_l2_regularizer(),
            name=prefix + 'expand')(x)
        x = tf.keras.layers.BatchNormalization(
            epsilon=BATCH_NORM_EPSILON,
            momentum=BATCH_NORM_DECAY,
            name=prefix + 'expand_BN')(x, training=is_training)
        x = tf.keras.layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = tf.keras.layers.ZeroPadding2D(
            padding=((0, 1), (0, 1)),
            name=prefix + 'pad')(x)
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        activation=None,
        use_bias=False,
        padding='same' if stride == 1 else 'valid',
        kernel_initializer=_gen_initializer(),
        kernel_regularizer=_gen_l2_regularizer(False),
        name=prefix + 'depthwise')(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=BATCH_NORM_EPSILON,
        momentum=BATCH_NORM_DECAY,
        name=prefix + 'depthwise_BN')(x, training=is_training)

    x = tf.keras.layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = tf.keras.layers.Conv2D(
        pointwise_filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        activation=None,
        kernel_initializer=_gen_initializer(),
        kernel_regularizer=_gen_l2_regularizer(),
        name=prefix + 'project')(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=BATCH_NORM_EPSILON,
        momentum=BATCH_NORM_DECAY,
        name=prefix + 'project_BN')(x, training=is_training)

    if in_channels == pointwise_filters and stride == 1:
        return tf.keras.layers.Add(name=prefix + 'add')([inputs, x])
    return x


def _make_divisible(value, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value
