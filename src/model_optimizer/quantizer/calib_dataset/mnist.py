# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mnist dataset
"""

import tensorflow as tf
from .dataset_base import DatasetBase


class MnistDataset(DatasetBase):
    """
    Mnist dataset
    """

    def __init__(self, data_path):
        """
        Constructor function.
        :param data_path: tfrecord data path
        :return:
        """
        super().__init__(data_path)
        self.dataset_fn = tf.data.TFRecordDataset

    # pylint: disable=R0201
    def parse_fn(self, example_serialized):
        """
        Parse features from the serialized data
        :param example_serialized: serialized data
        :return: image,label
        """
        feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        features = tf.io.parse_single_example(example_serialized, feature_description)
        image = tf.io.decode_raw(features['image_raw'], tf.uint8)
        image = tf.cast(image, dtype='float32') / 255.0
        label = tf.cast(features['label'], dtype=tf.int32)
        return tf.reshape(image, [28, 28, 1]), label
