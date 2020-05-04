# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Model quantizer
"""


def create_quantizer(config, calibrate_input_func):
    """
    Get model quantizer
    :param config: Config object
    :return: class of quantizer
    """
    model_type = config.get_attribute('model_type')
    if model_type not in ['tflite', 'tftrt']:
        raise Exception('Not support model type %s' % model_type)
    if model_type == 'tflite':
        from .tflite.optimizer import Quantizer
        return Quantizer(config, calibrate_input_func)
    elif model_type == 'tftrt':
        from .tftrt.optimizer import Quantizer
        return Quantizer(config, calibrate_input_func)
    else:
        raise Exception('Not support platform {}'.format(model_type))