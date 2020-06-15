# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pruner class define and function related
"""
import networkx as nx
import tensorflow as tf
import numpy as np


def get_network(model):
    """
    Get networkx's graph from model
    :param model: keras functional Model
    :return: networkx's DiGraph
    """
    digraph = nx.DiGraph()
    for i, layer in enumerate(model.layers):
        digraph.add_node(i, name=layer.name, type=str(type(layer)))
    for i, layer in enumerate(model.layers):
        for j in range(0, len(model.layers)):
            _inputs = model.layers[j].input
            if isinstance(_inputs, list):
                for _input in _inputs:
                    if layer.output.name == _input.name:
                        digraph.add_edge(i, j, name=layer.input.name)
            elif layer.output.name == _inputs.name:
                digraph.add_edge(i, j, name=layer.output.name)
    return digraph


def _get_sorted_mask(arr, num_retain_channels):
    arg_sort = np.argsort(-arr)
    mask_arr = np.zeros(arr.shape[-1], dtype=bool)
    mask_arr[arg_sort[:num_retain_channels]] = True
    return mask_arr


def _get_conv_mask(model, layer_id, digraph, num_retain_channels, criterion):
    return _get_layer_mask(model, layer_id, 'conv', digraph, num_retain_channels, criterion)


def _get_dense_mask(model, layer_id, digraph, num_retain_channels, criterion):
    return _get_layer_mask(model, layer_id, 'dense', digraph, num_retain_channels, criterion)


def _get_layer_mask(model, layer_id, layer_type, digraph, num_retain_channels, criterion):
    """
    Get conv2d layer mask
    :param model: keras model
    :param layer_id: keras model layer index
    :param layer_type: 'dense' or 'conv'
    :param digraph: networkx's DiGraph
    :param num_retain_channels: number of channels retained
    :param criterion: 'l1_norm' 'bn_gamma'
    :return: conv2d layer mask
    """
    layer = model.layers[layer_id]
    if criterion == 'l1_norm':
        kernel = layer.weights[0].numpy()
        if layer_type == 'conv':
            l1_norm = np.sum(np.abs(kernel), axis=(0, 1, 2))
        else:
            l1_norm = np.sum(np.abs(kernel), axis=(0,))
        arr = l1_norm
    elif criterion == 'bn_gamma':
        next_ids = digraph.successors(layer_id)
        for idx in next_ids:
            next_id = idx
        if digraph.nodes[next_id]['type'].endswith('BatchNormalization\'>'):
            bn_layer = model.layers[next_id]
            gamma = bn_layer.weights[0].numpy()
            arr = gamma
        else:
            kernel = layer.weights[0].numpy()
            if layer_type == 'conv':
                l1_norm = np.sum(np.abs(kernel), axis=(0, 1, 2))
            else:
                l1_norm = np.sum(np.abs(kernel), axis=(0,))
            arr = l1_norm
    return _get_sorted_mask(arr, num_retain_channels)


def get_relate_father_id(layer_id, digraph):
    """
    Get relate layer index
      for example:
      conv1-->avgpool2d-->flatten-->dense

      when prune dense, must get dense mask together with conv1 layer mask
    :param layer_id: keras Model layer index
    :param digraph: networkx's DiGraph
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
    for node in digraph.predecessors(layer_id):
        node_list.append(node)
    if len(node_list) >= 1:
        new_id = node_list[0]
        if digraph.nodes[new_id]['type'].split(sep='.')[-1][:-2] in skip_op:
            relate_id = get_relate_father_id(new_id, digraph)
        elif digraph.nodes[new_id]['type'].endswith('Conv2D\'>') or digraph.nodes[new_id]['type'].endswith('Dense\'>'):
            relate_id = new_id
        else:
            relate_id = -1
    return relate_id


def _bn_layer_set_weights(pruned_model, layer, idx, mask_father_id, mask_dict):
    if mask_father_id in mask_dict:
        weights_gamma = layer.weights[0].numpy()[mask_dict[mask_father_id]]
        weights_beta = layer.weights[1].numpy()[mask_dict[mask_father_id]]
        weights_moving_mean = layer.weights[2].numpy()[mask_dict[mask_father_id]]
        weights_moving_variance = layer.weights[3].numpy()[mask_dict[mask_father_id]]
        pruned_model.layers[idx].set_weights([weights_gamma, weights_beta, weights_moving_mean,
                                              weights_moving_variance])
    else:
        pruned_model.layers[idx].set_weights(layer.get_weights())


def _layer_set_weights(pruned_model, layer, weights_0, idx, mask_dict):
    layer_type = str(type(layer))
    if idx in mask_dict:
        if layer_type.endswith('Conv2D\'>'):
            if layer.use_bias:
                pruned_model.layers[idx].set_weights([weights_0[:, :, :, mask_dict[idx]],
                                                      layer.weights[1].numpy()[mask_dict[idx]]])
            else:
                pruned_model.layers[idx].set_weights([weights_0[:, :, :, mask_dict[idx]]])
        else:
            if layer.use_bias:
                pruned_model.layers[idx].set_weights([weights_0[:, mask_dict[idx]],
                                                      layer.weights[1].numpy()[mask_dict[idx]]])
            else:
                pruned_model.layers[idx].set_weights(
                    [weights_0[:, mask_dict[idx]]])
    else:
        if layer.use_bias:
            pruned_model.layers[idx].set_weights([weights_0, layer.get_weights()[1]])
        else:
            pruned_model.layers[idx].set_weights([weights_0])


def update_weights(model, pruned_model, digraph, mask_dict):
    """
    Update all weights with pruned mask dict
    :param model: model before pruned
    :param pruned_model: model after pruned
    :param digraph: networkx's DiGraph
    :param mask_dict: mask dict which save all mask for all layers to be pruned
    :return:
    """
    for i, layer in enumerate(model.layers):
        layer_type = str(type(layer))
        if layer_type.endswith('Conv2D\'>') or layer_type.endswith('Dense\'>') or \
                layer_type.endswith('BatchNormalization\'>'):
            new_model_layer_input_shape = pruned_model.layers[i].input.shape
            model_layer_input_shape = layer.input.shape
            mask_father_id = -1
            if new_model_layer_input_shape != model_layer_input_shape:
                mask_father_id = get_relate_father_id(i, digraph)
            if layer_type.endswith('BatchNormalization\'>'):
                _bn_layer_set_weights(pruned_model, layer, i, mask_father_id, mask_dict)
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
            _layer_set_weights(pruned_model, layer, weights_0, i, mask_dict)


def specified_layers_prune(orig_model, cur_model, layers_name, ratio, criterion='l1_norm'):
    """
    Prune with specified layers
    :param orig_model: original model, never pruned once
    :param cur_model: model before this step of pruned
    :param layers_name: name list of pruned layers
    :param ratio: ratio of pruned
    :param criterion: 'l1_norm' or 'bn_gamma'
    :return: pruned model
    """
    clone_model = tf.keras.models.clone_model(cur_model)
    digraph = get_network(cur_model)
    mask_dict = {}
    for i, layer in enumerate(cur_model.layers):
        if 'Conv2D' in str(type(layer)):
            if layer.name in layers_name:
                clone_model.layers[i].filters = \
                    clone_model.layers[i].filters - int(orig_model.layers[i].filters * ratio)
                mask_dict[i] = _get_conv_mask(cur_model, i, digraph, int(clone_model.layers[i].filters), criterion)
        elif 'Dense' in str(type(layer)):
            if i == len(cur_model.layers) - 1:
                continue
            else:
                if layer.name in layers_name:
                    clone_model.layers[i].units = \
                        clone_model.layers[i].units - int(orig_model.layers[i].units * ratio)
                    mask_dict[i] = _get_dense_mask(cur_model, i, digraph, int(clone_model.layers[i].units), criterion)
    pruned_model = tf.keras.models.model_from_json(clone_model.to_json())
    update_weights(cur_model, pruned_model, digraph, mask_dict)
    return pruned_model


def auto_prune(orig_model, cur_model, ratio, criterion='l1_norm'):
    """
    Auto prune layer with fixed ratio
    :param orig_model: original model, never pruned once
    :param cur_model: model before this step of pruned
    :param ratio: ratio of pruned
    :param criterion: 'l1_norm' or 'bn_gamma'
    :return: pruned model
    """
    clone_model = tf.keras.models.clone_model(cur_model)
    digraph = get_network(cur_model)
    mask_dict = {}
    for i, layer in enumerate(cur_model.layers):
        if 'Conv2D' in str(type(layer)):
            clone_model.layers[i].filters = \
                clone_model.layers[i].filters - int(orig_model.layers[i].filters * ratio)
            mask_dict[i] = _get_conv_mask(cur_model, i, digraph, int(clone_model.layers[i].filters), criterion)
        elif 'Dense' in str(type(layer)):
            if i == len(cur_model.layers) - 1:
                continue
            else:
                clone_model.layers[i].units = \
                    clone_model.layers[i].units - int(orig_model.layers[i].units * ratio)
                mask_dict[i] = _get_dense_mask(cur_model, i, digraph, int(clone_model.layers[i].units), criterion)
    pruned_model = tf.keras.models.model_from_json(clone_model.to_json())
    update_weights(cur_model, pruned_model, digraph, mask_dict)
    return pruned_model


class AutoPruner:
    """
    Auto select layers to prune.
    """

    def __init__(self, config):
        self.ratio = config['ratio']
        self.criterion = config['criterion']

    def prune(self, orig_model, cur_model):
        """
        Auto prune layer with fixed ratio
        :param orig_model: original model, never pruned once
        :param cur_model: model before this step of pruned
        :return: pruned model
        """
        return auto_prune(orig_model, cur_model, self.ratio, self.criterion)


class SpecifiedLayersPruner:
    """
    Specified layers to prune.
    """

    def __init__(self, config):
        self.ratio = config['ratio']
        self.criterion = config['criterion']
        self.layers_name = config['layers_to_be_pruned']

    def prune(self, orig_model, cur_model):
        """
        Prune with specified layers
        :param orig_model: original model, never pruned once
        :param cur_model: model before this step of pruned
        :return: pruned model
        """
        return specified_layers_prune(orig_model, cur_model, self.layers_name, self.ratio, self.criterion)
