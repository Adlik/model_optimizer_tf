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

    def __init__(self, data_path):
        self.data_path = data_path
        _LOGGER.info('data_dir is: %s', self.data_path)

    def input_gen(self):
        dataset = self.__build()
        for image, _ in dataset:
            yield [image]

    def __build(self):
        dataset = tf.data.Dataset.list_files(self.data_path, shuffle=True)
        dataset = dataset.interleave(self.dataset_fn, cycle_length=10, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(1)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset


