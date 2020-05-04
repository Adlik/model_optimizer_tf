# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
dataset base class
"""

import tensorflow as tf
from ...log_util import get_logger
_LOGGER = get_logger(__name__)


class DatasetBase:
    """
    Dataset base class
    """
    _COMMON_PARAMS = [
        "data_dir",
        "model_name",
        "export_path",
        "version",
        "batch_size",
        "batch_size_val"
    ]
    _COMMON_REQUIRED = [
        "data_dir",
        "batch_size",
        "batch_size_val"
    ]

    def __init__(self, config, is_training, shard=[1, 0]):
        for item in self._COMMON_PARAMS:
            if config.get_attribute(item) is None and item in self._COMMON_REQUIRED:
                _LOGGER.error('Require "%s" but not found', item)
                raise Exception('Require "%s" but not found' % item)
            self.__setattr__(item, config.get_attribute(item))

        _LOGGER.info('data_dir is: %s', self.data_dir)
        self.is_training = is_training
        self.shard = shard

    @property
    def steps_per_epoch(self):
        if self.is_training:
            return int(self.num_samples_of_train/self.batch_size)
        else:
            return int(self.num_samples_of_val / self.batch_size_val)

    @property
    def num_samples(self):
        if self.is_training:
            return self.num_samples_of_train
        else:
            return self.num_samples_of_val

    def build(self, split=0.0):
        dataset = tf.data.Dataset.list_files(self.file_pattern, shuffle=True)
        if self.shard[0] != 1:
            dataset = dataset.shard(num_shards=self.shard[0], index=self.shard[1])
        dataset = dataset.interleave(self.dataset_fn, cycle_length=10, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.is_training:
            dataset = dataset.shuffle(buffer_size=self.buffer_size).repeat()
        dataset = dataset.map(self.parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # create iterators for training & validation subsets separately
        if self.is_training and split > 0:
            nb_samples_val = int(self.read_tfrecord*split)
            batch_val = self.__build_batch(dataset.take(nb_samples_val))
            batch_train = self.__build_batch(dataset.skip(nb_samples_val))
            return batch_train, batch_val

        return self.__build_batch(dataset)

    def __build_batch(self, dataset):
        """
        Make an batch from tf.data.Dataset.
        :param dataset: tf.data.Dataset object
        :return: an batch of dataset
        """
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

