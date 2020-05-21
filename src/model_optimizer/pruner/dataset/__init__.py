# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dataset
"""


def get_dataset(config, is_training, num_shards=1, shard_index=0):
    """
    Get dataset
    :param config: Config object
    :param is_training: get training dataset or val dataset
    :param num_shards: dataset split into shard[0] part
    :param shard_index: get subset of dataset with index shard_index,
    :return: class of Dataset
    """
    dataset_name = config.get_attribute('dataset')
    if dataset_name not in ['mnist', 'cifar10', 'imagenet']:
        raise Exception('Not support dataset %s' % dataset_name)
    if dataset_name == 'mnist':
        from .mnist import MnistDataset
        return MnistDataset(config, is_training)
    elif dataset_name == 'cifar10':
        from .cifar10 import Cifar10Dataset
        return Cifar10Dataset(config, is_training)
    elif dataset_name == 'imagenet':
        from .imagenet import ImagenetDataset
        return ImagenetDataset(config, is_training, num_shards, shard_index)
    else:
        raise Exception('Not support dataset {}'.format(dataset_name))
