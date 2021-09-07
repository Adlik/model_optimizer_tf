# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Get model
"""
from ..scheduler.common import get_scheduler
from ..distill.distiller import get_distiller


# pylint: disable=too-many-return-statements
# pylint: disable=too-many-branches
def get_model(config, input_shape, is_training=True):
    """
    Get model
    :param config: Config object
    :param is_training: if training or not
    :return: class of keras Model
    """
    model_name = config.get_attribute('model_name')
    scheduler_config = get_scheduler(config)
    if model_name not in ['lenet', 'resnet_18', 'vgg_m_16', 'resnet_50', 'resnet_101',
                          'mobilenet_v1', 'mobilenet_v2', 'cnn1d', 'cnn1d_tiny']:
        raise Exception('Not support model %s' % model_name)

    if (config.get_attribute('scheduler') == 'distill' or config.get_attribute('is_distill')) and is_training:
        classifier_activation = None
    else:
        classifier_activation = 'softmax'
    if model_name == 'lenet':
        from .lenet import lenet
        student_model = lenet(is_training, model_name, classifier_activation=classifier_activation)
    elif model_name == 'vgg_m_16':
        from .vgg import vgg_m_16
        student_model = vgg_m_16(is_training, model_name, classifier_activation=classifier_activation)
    elif model_name == 'resnet_18':
        from .resnet import resnet_18
        student_model = resnet_18(is_training, model_name, classifier_activation=classifier_activation)
    elif model_name == 'resnet_50':
        from .resnet import resnet_50
        student_model = resnet_50(is_training, model_name, classifier_activation=classifier_activation)
    elif model_name == 'resnet_101':
        from .resnet import resnet_101
        student_model = resnet_101(is_training, model_name, classifier_activation=classifier_activation)
    elif model_name == 'mobilenet_v1':
        from .mobilenet_v1 import mobilenet_v1_1
        student_model = mobilenet_v1_1(is_training=is_training, name=model_name,
                                       classifier_activation=classifier_activation)
    elif model_name == 'mobilenet_v2':
        from .mobilenet_v2 import mobilenet_v2_1
        student_model = mobilenet_v2_1(is_training=is_training, name=model_name,
                                       classifier_activation=classifier_activation)
    elif model_name == 'cnn1d':
        from .cnn1d import cnn1d
        student_model = cnn1d(is_training=is_training, name=model_name, classifier_activation=classifier_activation)
    elif model_name == 'cnn1d_tiny':
        from .cnn1d import cnn1d_tiny
        student_model = cnn1d_tiny(is_training=is_training, name=model_name,
                                   classifier_activation=classifier_activation)
    if (config.get_attribute('scheduler') == 'distill' or config.get_attribute('is_distill')) and is_training:
        distill_model = get_distiller(student_model, scheduler_config, input_shape)
    else:
        distill_model = student_model
    return distill_model
