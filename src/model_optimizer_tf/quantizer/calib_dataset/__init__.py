# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dataset input function
"""


def input_fn(dataset_name, data_path):
    """
    Get dataset
    :param dataset_name: dataset name
    :param data_path: tfrecord data path
    :return: func of input
    """
    if dataset_name not in ['mnist', 'cifar10', 'imagenet']:
        raise Exception(f'Not support dataset {dataset_name}')
    if dataset_name == 'mnist':
        from .mnist import MnistDataset
        return MnistDataset(data_path).input_gen
    elif dataset_name == 'cifar10':
        from .cifar10 import Cifar10Dataset
        return Cifar10Dataset(data_path).input_gen
    elif dataset_name == 'imagenet':
        from .imagenet import ImagenetDataset
        return ImagenetDataset(data_path).input_gen
    else:
        raise Exception(f'Not support dataset {dataset_name}')
