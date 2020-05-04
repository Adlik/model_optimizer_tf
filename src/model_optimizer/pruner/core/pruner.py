# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import networkx as nx
import tensorflow as tf
import numpy as np


def get_network(model):
    """
    Get networkx's graph from model
    :param model: keras functional Model
    :return: networkx's DiGraph
    """
    g = nx.DiGraph()
    for i, layer in enumerate(model.layers):
        g.add_node(i, name=layer.name, type=str(type(layer)))
    for i, layer in enumerate(model.layers):
        for j in range(0, len(model.layers)):
            _inputs = model.layers[j].input
            if isinstance(_inputs, list):
                for _input in _inputs:
                    if layer.output.name == _input.name:
                        g.add_edge(i, j, name=layer.input.name)
            elif layer.output.name == _inputs.name:
                g.add_edge(i, j, name=layer.output.name)
    return g


def get_conv_mask(layer, num_retain_channels):
    """
    Get conv2d layer mask
    :param layer: keras Model layer
    :param num_retain_channels: number of channels retained
    :return: conv2d layer mask
    """
    kernel = layer.weights[0].numpy()
    l1_norm = np.sum(np.abs(kernel), axis=(0, 1, 2))
    arg_sort = np.argsort(-l1_norm)
    mask_kernel = np.zeros(kernel.shape[-1], dtype=bool)
    mask_kernel[arg_sort[:num_retain_channels]] = True
    return mask_kernel


def get_dense_mask(layer, num_retain_channels):
    """
    Get dense layer mask
    :param layer: keras Model layer
    :param num_retain_channels: number of channels retained
    :return: dense layer mask
    """
    kernel = layer.weights[0].numpy()
    l1_norm = np.sum(np.abs(kernel), axis=(0,))
    arg_sort = np.argsort(-l1_norm)
    mask_kernel = np.zeros(kernel.shape[-1], dtype=bool)
    mask_kernel[arg_sort[:num_retain_channels]] = True
    return mask_kernel


def get_relate_father_id(layer_id, g):
    """
    Get relate layer index
      for example:
      conv1-->avgpool2d-->flatten-->dense

      when prune dense, must get dense mask together with conv1 layer mask
    :param layer_id: keras Model layer index
    :param g: networkx's DiGraph
    :return: relate layer id
    """
    relate_id = -1
    skip_op = ['Activation',
               'Add',
               'AveragePooling2D',
               'BatchNormalization',
               'Flatten',
               'MaxPooling2D']
    node_list = []
    for node in g.predecessors(layer_id):
        node_list.append(node)
    if len(node_list) >= 1:
        new_id = node_list[0]
        if g.nodes[new_id]['type'].split(sep='.')[-1][:-2] in skip_op:
            relate_id, _ = get_relate_father_id(new_id, g)
        elif g.nodes[new_id]['type'].endswith('Conv2D\'>') or g.nodes[new_id]['type'].endswith('Dense\'>'):
            relate_id = new_id
        else:
            relate_id = -1
    return relate_id, g


def update_weights(model, pruned_model, mask_dict):
    """
    Update all weights with pruned mask dict
    :param model: model before pruned
    :param pruned_model: model after pruned
    :param mask_dict: mask dict which save all mask for all layers to be pruned
    :return:
    """
    g = get_network(model)
    for i, layer in enumerate(model.layers):
        layer_type = str(type(layer))
        if layer_type.endswith('Conv2D\'>') or layer_type.endswith('Dense\'>') or \
                layer_type.endswith('BatchNormalization\'>'):
            new_model_layer_input_shape = pruned_model.layers[i].input.shape
            model_layer_input_shape = layer.input.shape
            mask_father_id = -1
            if new_model_layer_input_shape != model_layer_input_shape:
                mask_father_id, _ = get_relate_father_id(i, g)
            if layer_type.endswith('BatchNormalization\'>'):
                if mask_father_id in mask_dict:
                    weights_gamma = layer.weights[0].numpy()[mask_dict[mask_father_id]]
                    weights_beta = layer.weights[1].numpy()[mask_dict[mask_father_id]]
                    weights_moving_mean = layer.weights[2].numpy()[mask_dict[mask_father_id]]
                    weights_moving_variance = layer.weights[3].numpy()[mask_dict[mask_father_id]]
                    pruned_model.layers[i].set_weights([weights_gamma, weights_beta, weights_moving_mean,
                                                        weights_moving_variance])
                else:
                    pruned_model.layers[i].set_weights(layer.get_weights())
                continue
            if mask_father_id != -1 and mask_father_id in mask_dict:
                weights_0 = layer.weights[0].numpy().reshape(
                    -1,
                    model.layers[mask_father_id].output.shape[-1],
                    layer.weights[0].numpy().shape[-1])[:, mask_dict[mask_father_id], :]
                if layer_type.endswith('Conv2D\'>'):
                    weights_0 = weights_0.reshape(layer.weights[0].shape[0], layer.weights[0].shape[1], -1,
                                                  layer.weights[0].shape[-1])
                else:
                    weights_0 = weights_0.reshape(-1, weights_0.shape[-1])
            else:
                weights_0 = layer.weights[0].numpy()
            if i in mask_dict:
                if layer_type.endswith('Conv2D\'>'):
                    if layer.use_bias:
                        pruned_model.layers[i].set_weights([weights_0[:, :, :, mask_dict[i]],
                                                            layer.weights[1].numpy()[mask_dict[i]]])
                    else:
                        pruned_model.layers[i].set_weights([weights_0[:, :, :, mask_dict[i]]])
                else:
                    if layer.use_bias:
                        pruned_model.layers[i].set_weights(
                            [weights_0[:, mask_dict[i]], layer.weights[1].numpy()[mask_dict[i]]])
                    else:
                        pruned_model.layers[i].set_weights(
                            [weights_0[:, mask_dict[i]]])
            else:
                if layer.use_bias:
                    pruned_model.layers[i].set_weights([weights_0, layer.get_weights()[1]])
                else:
                    pruned_model.layers[i].set_weights([weights_0])


def specified_layers_prune(model, layers_name, ratio):
    """
    Prune with specified layers
    :param model: model before pruned
    :param layers_name: name list of pruned layers
    :param ratio: ratio of pruned
    :return: pruned model
    """
    g = get_network(model)
    clone_model = tf.keras.models.clone_model(model)
    mask_dict = {}
    for i, layer in enumerate(model.layers):
        if 'Conv2D' in str(type(layer)):
            if layer.name in layers_name:
                clone_model.layers[i].filters = int(clone_model.layers[i].filters * (1 - ratio))
                mask_dict[i] = get_conv_mask(layer, int(clone_model.layers[i].filters))
        elif 'Dense' in str(type(layer)):
            if i == len(model.layers) - 1:
                continue
            else:
                if layer.name in layers_name:
                    clone_model.layers[i].units = int(clone_model.layers[i].units * (1 - ratio))
                    mask_dict[i] = get_dense_mask(layer, int(clone_model.layers[i].units))
    pruned_model = tf.keras.models.model_from_json(clone_model.to_json())
    update_weights(model, pruned_model, mask_dict)
    return pruned_model


def auto_prune(model, ratio):
    """
    Auto prune layer with fixed ratio
    :param model: model before pruned
    :param ratio: ratio of pruned
    :return: pruned model
    """
    clone_model = tf.keras.models.clone_model(model)
    mask_dict = {}
    for i, layer in enumerate(model.layers):
        if 'Conv2D' in str(type(layer)):
            clone_model.layers[i].filters = int(clone_model.layers[i].filters*(1-ratio))
            mask_dict[i] = get_conv_mask(layer, int(clone_model.layers[i].filters))
        elif 'Dense' in str(type(layer)):
            if i == len(model.layers) - 1:
                continue
            else:
                clone_model.layers[i].units = int(clone_model.layers[i].units * (1-ratio))
                mask_dict[i] = get_dense_mask(layer, int(clone_model.layers[i].units))
    pruned_model = tf.keras.models.model_from_json(clone_model.to_json())
    update_weights(model, pruned_model, mask_dict)
    return pruned_model


class AutoPruner(object):
    """
    Auto select layers to prune.
    """

    def __init__(self, config):
        self.ratio = config['ratio']
        self.criterion = config['criterion']

    def prune(self, model):
        return auto_prune(model, self.ratio)


class SpecifiedLayersPruner(object):
    """
    Specified layers to prune.
    """

    def __init__(self, config):
        self.ratio = config['ratio']
        self.criterion = config['criterion']
        self.layers_name = config['layers_to_be_pruned']

    def prune(self, model):
        return specified_layers_prune(model, self.layers_name, self.ratio)

