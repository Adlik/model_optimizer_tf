# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Learner
"""


def get_learner(config):
    """
    Get Learner
    :param config: Config object
    :return: class of Learner
    """
    model_name = config.get_attribute('model_name')
    dataset_name = config.get_attribute('dataset')
    if model_name == 'lenet' and dataset_name == 'mnist':
        from .lenet_mnist import Learner
        return Learner(config)
    elif model_name == 'resnet_50' and dataset_name == 'imagenet':
        from .resnet_50_imagenet import Learner
        return Learner(config)
    elif model_name == 'resnet_101' and dataset_name == 'imagenet':
        from .resnet_101_imagenet import Learner
        return Learner(config)
    elif model_name == 'mobilenet_v1' and dataset_name == 'imagenet':
        from .mobilenet_v1_imagenet import Learner
        return Learner(config)
    elif model_name == 'mobilenet_v2' and dataset_name == 'imagenet':
        from .mobilenet_v2_imagenet import Learner
        return Learner(config)
    elif model_name == 'vgg_m_16' and dataset_name == 'cifar10':
        from .vgg_m_16_cifar10 import Learner
        return Learner(config)
    else:
        raise Exception('Not support learner: {}_{}'.format(model_name, dataset_name))
