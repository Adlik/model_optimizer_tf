# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Params and FLOPs
"""
import tensorflow as tf
import numpy as np


# pylint: disable=not-context-manager
def get_keras_model_params_flops(model_h5_path):
    """
    Get keras model FLOPs
    :param model_h5_path: keras model path
    :return: Params, FLOPs
    """
    model = tf.keras.models.load_model(model_h5_path)
    total_params, total_flops = _count_model_params_flops(model)
    return total_params, total_flops


def print_keras_model_summary(model, hvd_rank):
    """
    Print keras model summary
    :param model: keras model
    :param hvd_rank: horovod rank
    :return:
    """
    if hvd_rank != 0:
        return
    print(model.summary())


def print_keras_model_params_flops(model, hvd_rank):
    """
    Print keras model paras and FLOPs
    :param model: keras model
    :param hvd_rank: horovod rank
    :return:
    """
    if hvd_rank != 0:
        return
    total_params, total_flops = _count_model_params_flops(model)
    print(f'total params: {str(total_params)}')
    print(f'total flops: {str(total_flops)}')


def _count_conv_layer_flops(conv_layer):
    out_shape = conv_layer.output.shape.as_list()
    n_cells_total = np.prod(out_shape[1:-1])
    n_conv_params_total = conv_layer.count_params()
    conv_flops = n_conv_params_total * n_cells_total
    return conv_flops


def _count_dense_layer_flops(dense_layer):
    out_shape = dense_layer.output.shape.as_list()
    in_shape = dense_layer.input.shape.as_list()
    dense_flops = np.prod([out_shape[-1], in_shape[-1]])
    return dense_flops


def _count_model_params_flops(model):
    total_params = 0
    total_flops = 0
    for layer in model.layers:
        total_params += layer.count_params()
        if any(conv_type in str(type(layer)) for conv_type in ['Conv1D', 'Conv2D', 'Conv3D']):
            flops = _count_conv_layer_flops(layer)
            total_flops += flops
        elif 'Dense' in str(type(layer)):
            flops = _count_dense_layer_flops(layer)
            total_flops += flops
        else:
            print(f'warning:: skippring layer: {str(layer)}')
    return total_params, total_flops
