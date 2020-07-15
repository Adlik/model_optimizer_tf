# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Get model
"""


def get_model(config, is_training=True):
    """
    Get model
    :param config: Config object
    :param is_training: if training or not
    :return: class of keras Model
    """
    model_name = config.get_attribute('model_name')
    if model_name not in ['lenet', 'resnet_18', 'vgg_m_16', 'resnet_50',
                          'mobilenet_v1', 'mobilenet_v2']:
        raise Exception('Not support model %s' % model_name)
    if model_name == 'lenet':
        from .lenet import lenet
        return lenet(is_training)
    elif model_name == 'vgg_m_16':
        from .vgg import vgg_m_16
        return vgg_m_16(is_training)
    elif model_name == 'resnet_18':
        from .resnet import resnet_18
        return resnet_18(is_training)
    elif model_name == 'resnet_50':
        from .resnet import resnet_50
        return resnet_50(is_training)
    elif model_name == 'mobilenet_v1':
        from .mobilenet_v1 import mobilenet_v1_1
        return mobilenet_v1_1(is_training=is_training)
    elif model_name == 'mobilenet_v2':
        from .mobilenet_v2 import mobilenet_v2_1
        return mobilenet_v2_1(is_training=is_training)
    else:
        raise Exception('Not support model {}'.format(model_name))
