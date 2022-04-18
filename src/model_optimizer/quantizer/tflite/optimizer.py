# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TF-Lite quantizer
"""
import os
import tensorflow as tf
from ..quantizer_base import BaseQuantizer
from ...log_util import get_logger

_LOGGER = get_logger(__name__)


class Quantizer(BaseQuantizer):
    """
    SavedModel quantizer
    """

    def __init__(self, config, calibration_input_fn):
        super().__init__(config)
        self.calibration_input_fn = calibration_input_fn

    def _do_quantize(self):
        """
        convert SavedModel to tflite model
        :return: Return convert result
        """
        _LOGGER.info('Start to convert tflite model')
        if self.input_model.endswith('.h5'):
            keras_model = tf.keras.models.load_model(self.input_model)
            converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        else:
            converter = tf.lite.TFLiteConverter.from_saved_model(self.input_model)
        converter.representative_dataset = self.calibration_input_fn
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_quant_model = converter.convert()
        with open(os.path.join(self.target_dir, self.model_name+".tflite"), "wb") as quan_file:
            quan_file.write(tflite_quant_model)

    @staticmethod
    def _write_version_file(saved_model_path):
        version_path = os.path.dirname(saved_model_path)
        if isinstance(version_path, bytes):
            version_path = version_path.decode('utf-8')
        # pylint: disable=unspecified-encoding
        with open(os.path.join(str(version_path), "TFVERSION"), 'w') as version_file:
            version_file.write(tf.__version__)
        _LOGGER.info('Write TFVERSION file success, path: %s', version_path)

    @staticmethod
    def get_platform():
        """
        Get platform
        :return:
        """
        return "tensorflow", tf.__version__
