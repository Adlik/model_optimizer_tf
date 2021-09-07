"""
1D-CNN model
"""
import tensorflow as tf


def cnn1d(is_training=True, name='cnn1d', classifier_activation='softmax'):
    """
    This implements a 1D-CNN by Wei Wang
    [Wang, W.; Zhu, M.; Wang, J.; Zeng, X.; Yang, Z. End-to-end encrypted traffic classification with one-dimensional
    convolution neural networks.]
    :param is_training: if training or not
    :param name: the model name
    :param classifier_activation: classifier_activation can only be None or "softmax"
    :return: cnn1d model
    """
    input_ = tf.keras.layers.Input(shape=(1, 784, 1), name='input')
    x = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=(1, 25),
                               padding='same',
                               activation='relu',
                               name='conv2d_1')(input_)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 3), padding='same',name='pool_1')(x)
    x = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(1, 25),
                               padding='same',
                               activation='relu',
                               name='conv2d_2')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 3), padding='same', name='pool_2')(x)
    x = tf.keras.layers.Flatten(name='flatten')(x)
    x = tf.keras.layers.Dense(1024, activation='relu',
                              name='dense_1')(x)
    if is_training:
        x = tf.keras.layers.Dropout(0.5, name='dropout')(x)
    if classifier_activation == 'softmax':
        output_ = tf.keras.layers.Dense(12, activation='softmax', name='dense_2')(x)
    else:
        output_ = tf.keras.layers.Dense(12, activation=None, name='dense_2')(x)
    model = tf.keras.Model(input_, output_, name=name)
    return model


def cnn1d_tiny(is_training=True, name='cnn1d_tiny', classifier_activation='softmax'):
    """
    This implements a tiny 1D-CNN which is much smaller than teh 1D-CNN by Wei Wang
    :param is_training: if training or not
    :param name: the model name
    :param classifier_activation: classifier_activation can only be None or "softmax"
    :return: cnn1d_tiny model
    """
    input_ = tf.keras.layers.Input(shape=(1, 784, 1), name='input')
    x = tf.keras.layers.Conv2D(filters=16,
                               kernel_size=(1, 25),
                               padding='same',
                               activation='relu',
                               name='conv2d_1')(input_)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 3), padding='same',name='pool_1')(x)
    x = tf.keras.layers.Flatten(name='flatten')(x)
    x = tf.keras.layers.Dense(32, activation='relu',
                              name='dense_1')(x)
    if is_training:
        x = tf.keras.layers.Dropout(0.1, name='dropout')(x)  # 0.1 or 0.2 acc about 86%
    if classifier_activation == 'softmax':
        output_ = tf.keras.layers.Dense(12, activation='softmax', name='dense_2')(x)
    else:
        output_ = tf.keras.layers.Dense(12, activation=None, name='dense_2')(x)
    model = tf.keras.Model(input_, output_, name=name)
    return model
