"""
Tests for the model_optimizer package.
"""
import tensorflow as tf
from model_optimizer.pruner.core import AutoPruner
from model_optimizer.pruner.core import SpecifiedLayersPruner
import numpy as np


def test_uniform_auto_prune():
    """
    Test the AutoPruner prune function.
    """
    input_ = tf.keras.layers.Input(shape=(3, 3, 1), name='input')
    x = tf.keras.layers.Conv2D(filters=2,
                               kernel_size=(2, 2),
                               name='conv2d')(input_)
    x = tf.keras.layers.Flatten(name='flatten')(x)
    output_ = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(input_, output_)
    conv_layer = model.layers[1]
    conv_layer.set_weights(
        [np.array(
            [
                [
                    [[0.5, 1.0]],
                    [[0.5, 1.0]]
                ],
                [
                    [[0.5, 1.0]],
                    [[0.5, 1.0]]
                ]
            ]), conv_layer.weights[1].numpy()
         ])
    config = {'ratio': 0.5, 'criterion': 'l1_norm'}
    auto_pruner = AutoPruner(config)
    new_model = auto_pruner.prune(model, model)
    pruned_weight = np.array(
        [
            [
                [[1.0]],
                [[1.0]]
            ],
            [
                [[1.0]],
                [[1.0]]
            ]
        ])
    res = np.array_equal(pruned_weight, new_model.layers[1].weights[0].numpy())
    assert res


def test_uniform_specified_layer_prune():
    """
    Test the SpecifiedLayersPruner prune function.
    """
    input_ = tf.keras.layers.Input(shape=(3, 3, 1), name='input')
    x = tf.keras.layers.Conv2D(filters=2,
                               kernel_size=(2, 2),
                               name='conv2d')(input_)
    x = tf.keras.layers.Flatten(name='flatten')(x)
    output_ = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(input_, output_)
    conv_layer = model.layers[1]
    conv_layer.set_weights(
        [np.array(
            [
                [
                    [[0.5, 1.0]],
                    [[0.5, 1.0]]
                ],
                [
                    [[0.5, 1.0]],
                    [[0.5, 1.0]]
                ]
            ]), conv_layer.weights[1].numpy()
         ])
    config = {'ratio': 0.5, 'criterion': 'l1_norm', 'layers_to_be_pruned': ['conv2d']}
    specified_layer_pruner = SpecifiedLayersPruner(config)
    new_model = specified_layer_pruner.prune(model, model)
    pruned_weight = np.array(
        [
            [
                [[1.0]],
                [[1.0]]
            ],
            [
                [[1.0]],
                [[1.0]]
            ]
        ])
    res = np.array_equal(pruned_weight, new_model.layers[1].weights[0].numpy())
    assert res
