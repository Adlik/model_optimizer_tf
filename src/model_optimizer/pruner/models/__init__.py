# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Get model
"""
from ..scheduler.common import get_scheduler
from ..distill.distiller import get_distiller


# pylint: disable=too-many-return-statements
def get_model(config, is_training=True):
    """
    Get model
    :param config: Config object
    :param is_training: if training or not
    :return: class of keras Model
    """
    model_name = config.get_attribute('model_name')
    scheduler_config = get_scheduler(config)
    if model_name not in ['lenet', 'resnet_18', 'vgg_m_16', 'resnet_50', 'resnet_101',
                          'mobilenet_v1', 'mobilenet_v2']:
        raise Exception('Not support model %s' % model_name)
    if model_name == 'lenet':
        from .lenet import lenet
        return lenet(model_name, is_training)
    elif model_name == 'vgg_m_16':
        from .vgg import vgg_m_16
        return vgg_m_16(is_training, model_name)
    elif model_name == 'resnet_18':
        from .resnet import resnet_18
        return resnet_18(is_training, model_name)
    elif model_name == 'resnet_50':
        from .resnet import resnet_50
        student_model = resnet_50(is_training, model_name)
        if config.get_attribute('scheduler') == 'distill':
            distill_model = get_distiller(student_model, scheduler_config)
            return distill_model
        else:
            return student_model
    elif model_name == 'resnet_101':
        from .resnet import resnet_101
        return resnet_101(is_training, model_name)
    elif model_name == 'mobilenet_v1':
        from .mobilenet_v1 import mobilenet_v1_1
        return mobilenet_v1_1(is_training=is_training, name=model_name)
    elif model_name == 'mobilenet_v2':
        from .mobilenet_v2 import mobilenet_v2_1
        return mobilenet_v2_1(is_training=is_training, name=model_name)
    else:
        raise Exception('Not support model {}'.format(model_name))
