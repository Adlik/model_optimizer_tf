# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
load tf models func
"""
import tensorflow as tf


def load_tf_ensemble_model(model_path):
    """
    Load multiple models to form an integrated model
    :param model_path: Model path, multiple model paths are separated by commas
    :return:
    """
    return EnsembleModel(model_path)


def _load_model(model_path, prefix=None):
    if 'tf2' in model_path:
        from tf2cv.model_provider import get_model as tf2cv_get_model  # type: ignore
        _model = tf2cv_get_model(model_path.split('/')[-1].split('-')[0], pretrained=False, data_format="channels_last")
        _model.build(input_shape=(1, 224, 224, 3))
        _model.load_weights(model_path)
        _model.trainable = False
    else:
        _model = tf.keras.models.load_model(model_path)
        _model.trainable = False
    if prefix is not None:
        for weight in _model.weights:
            weight._handle_name = prefix + '_' + weight.name  # pylint: disable=W0212
    return _model


class EnsembleModel(tf.keras.Model):  # pylint: disable=too-many-ancestors
    """
    Ensemble model for distillation
    """
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.avg = tf.keras.layers.Average()
        self.models = self._get_models(model_path)

    # pylint: disable=R0201
    def _get_models(self, model_path):
        models = []
        model_path_list = model_path.split(',')
        if len(model_path_list) == 1:
            _model = _load_model(model_path)
            models.append(_model)
        else:
            for i, _model_path in enumerate(model_path_list):
                _model = _load_model(_model_path, prefix='t'+str(i))
                models.append(_model)
        return models

    def call(self, inputs, training=None, mask=None):  # pylint: disable=unused-argument
        """
        Model call func
        :param inputs: Model input
        :return: average output
        """
        model_outputs = [model(inputs) for model in self.models]
        output = self.avg(model_outputs)
        return output

    def get_config(self):
        """
        Implement get_config to enable serialization.
        """
        config = super().get_config()
        config.update({"model_path": self.model_path})
        return config
