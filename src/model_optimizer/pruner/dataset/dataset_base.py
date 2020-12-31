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

    def __init__(self, config, is_training, num_shards=1, shard_index=0):
        for item in self._COMMON_PARAMS:
            if config.get_attribute(item) is None and item in self._COMMON_REQUIRED:
                _LOGGER.error('Require "%s" but not found', item)
                raise Exception('Require "%s" but not found' % item)
            self.__setattr__(item, config.get_attribute(item))

        _LOGGER.info('data_dir is: %s', self.data_dir)
        self.is_training = is_training
        self.num_shards = num_shards
        self.shard_index = shard_index

    @property
    def steps_per_epoch(self):
        """
        Steps per epoch
        :return: steps per epoch
        """
        if self.is_training:
            return int(self.num_samples_of_train/self.batch_size)
        else:
            return int(self.num_samples_of_val / self.batch_size_val)

    @property
    def num_samples(self):
        """
        Num of samples
        :return: num of samples
        """
        if self.is_training:
            return self.num_samples_of_train
        else:
            return self.num_samples_of_val

    def build(self, is_distill=False):
        """
        Build dataset
        :param is_distill: is distilling or not
        :return: batch of a dataset
        """
        dataset = tf.data.Dataset.list_files(self.file_pattern, shuffle=True)
        if self.num_shards != 1:
            dataset = dataset.shard(num_shards=self.num_shards, index=self.shard_index)
        dataset = dataset.interleave(self.dataset_fn, cycle_length=10, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.is_training:
            dataset = dataset.shuffle(buffer_size=self.buffer_size).repeat()
        if is_distill:
            dataset = dataset.map(self.parse_fn_distill, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.map(self.parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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
