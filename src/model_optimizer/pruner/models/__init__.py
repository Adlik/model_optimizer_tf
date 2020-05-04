# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Get model
"""


def get_model(config, is_training=True):
    """
    Get model
    :param config: Config object
    :return: class of keras Model
    """
    model_name = config.get_attribute('model_name')
    if model_name not in ['lenet', 'resnet_18', 'lenet_bn', 'resnet_50']:
        raise Exception('Not support dataset %s' % model_name)
    if model_name == 'lenet':
        from .lenet import lenet
        return lenet(is_training)
    elif model_name == 'lenet_bn':
        from .lenet_bn import lenet_bn
        return lenet_bn(is_training)
    elif model_name == 'resnet_18':
        from .resnet import resnet_18
        return resnet_18(is_training)
    elif model_name == 'resnet_50':
        from .resnet import resnet_50
        return resnet_50(is_training)
    else:
        raise Exception('Not support models {}'.format(model_name))