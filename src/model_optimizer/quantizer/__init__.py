# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Model quantizer
"""


def create_quantizer(config, calibrate_input_func):
    """
    Get model quantizer
    :param config: Config object
    :param calibrate_input_func: func to get input
    :return: class of quantizer
    """
    model_type = config.get_attribute('model_type')
    if model_type not in ['tflite', 'tftrt']:
        raise Exception(f'Not support model type {model_type}')
    if model_type == 'tflite':
        from .tflite.optimizer import Quantizer
        return Quantizer(config, calibrate_input_func)
    elif model_type == 'tftrt':
        from .tftrt.optimizer import Quantizer
        return Quantizer(config, calibrate_input_func)
    else:
        raise Exception(f'Not support platform {model_type}')
