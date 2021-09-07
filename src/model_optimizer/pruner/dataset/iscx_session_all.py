# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ISCX 12 class session all dataset
https://github.com/echowei/DeepTraffic/blob/master/2.encrypted_traffic_classification/3.PerprocessResults/12class.zip

"""
import os
import gzip
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file
import numpy as np
from .dataset_base import DatasetBase


class ISCXDataset(DatasetBase):
    """
    ISCX session all layer dataset
    """
    def __init__(self, config, is_training, num_shards=1, shard_index=0):
        """
        Constructor function.
        :param config: Config object
        :param is_training: whether to construct the training subset
        :return:
        """
        super().__init__(config, is_training, num_shards, shard_index)
        if is_training:
            self.batch_size = self.batch_size
        else:
            self.batch_size = self.batch_size_val
        self.buffer_size = 5000
        self.num_samples_of_train = 35501
        self.num_samples_of_val = 3945
        self.data_shape = (1, 784, 1)

    # pylint: disable=R0201
    # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
    def parse_fn(self, *content):
        data, label = content
        return data, label

    def parse_fn_distill(self, *content):
        """
        Parse dataset for distillation
        :param content: item content of the dataset
        :return: {image, label},{}
        """
        image, label = self.parse_fn(*content)
        inputs = {"image": image, "label": label}
        targets = {}
        return inputs, targets

    def build(self, is_distill=False):
        """
        Build dataset
        :param is_distill: is distilling or not
        :return: batch of a dataset
        """
        if self.is_training:
            x_path = os.path.join(self.data_dir, 'train-images-idx3-ubyte.gz')
            y_path = os.path.join(self.data_dir, 'train-labels-idx1-ubyte.gz')
        else:
            x_path = os.path.join(self.data_dir, 't10k-images-idx3-ubyte.gz')
            y_path = os.path.join(self.data_dir, 't10k-labels-idx1-ubyte.gz')

        with gzip.open(y_path, 'rb') as lbpath:
            y_data = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        with gzip.open(x_path, 'rb') as imgpath:
            x_data = np.frombuffer(
                imgpath.read(), np.uint8, offset=16).reshape(len(y_data), 1, 784)

        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

        if self.num_shards != 1:
            dataset = dataset.shard(num_shards=self.num_shards, index=self.shard_index)
        if self.is_training:
            dataset = dataset.shuffle(buffer_size=self.buffer_size).repeat()
        if is_distill:
            dataset = dataset.map(self.parse_fn_distill, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.map(self.parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self.build_batch(dataset)
