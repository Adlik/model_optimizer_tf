# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Cifar10 dataset.
"""
import os
import tensorflow as tf
from .dataset_base import DatasetBase


class Cifar10Dataset(DatasetBase):
    """
    Cifar10 dataset.
    """
    def __init__(self, config, is_training):
        """
        Constructor function.
        :param config: Config object
        :param is_training: whether to construct the training subset
        :return:
        """
        super().__init__(config, is_training)
        if is_training:
            self.file_pattern = os.path.join(self.data_dir, 'train.tfrecords')
            self.batch_size = self.batch_size
        else:
            self.file_pattern = os.path.join(self.data_dir, 'test.tfrecords')
            self.batch_size = self.batch_size_val
        self.dataset_fn = tf.data.TFRecordDataset
        self.buffer_size = 10000
        self.num_samples_of_train = 50000
        self.num_samples_of_val = 10000

    # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
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
        image = tf.reshape(image, [32, 32, 3])
        if self.is_training:
            image = tf.image.resize_with_crop_or_pad(image, 32 + 8, 32 + 8)
            image = tf.image.random_crop(image, [32, 32, 3])
            image = tf.image.random_flip_left_right(image)
        return image, label
