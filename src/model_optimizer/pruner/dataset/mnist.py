# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
from .dataset_base import DatasetBase
import os


class MnistDataset(DatasetBase):
    """
    mnist dataset
    """

    def __init__(self, config, is_training):
        """
        Constructor function.
        :param config: Config object
        :param is_training: whether to construct the training subset
        :return:
        """
        super(MnistDataset, self).__init__(config, is_training)
        if is_training:
            self.file_pattern = os.path.join(self.data_dir, 'train.tfrecords')
            self.batch_size = self.batch_size
        else:
            self.file_pattern = os.path.join(self.data_dir, 'test.tfrecords')
            self.batch_size = self.batch_size_val
        self.dataset_fn = tf.data.TFRecordDataset
        self.buffer_size = 10000
        self.num_samples_of_train = 60000
        self.num_samples_of_val = 10000

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



