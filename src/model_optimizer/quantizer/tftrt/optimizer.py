# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TF-TRT quantizer
"""
import os
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt  # pylint: disable=import-error,no-name-in-module
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
        convert SavedModel to tftrt SavedModel
        :return: Return convert result
        """
        _LOGGER.info('Start to convert SavedModel')
        if self.input_model.endswith('.h5'):
            keras_model = tf.keras.models.load_model(self.input_model)
            saved_model_path = os.path.join(os.path.dirname(self.input_model), 'saved_model')
            keras_model.save(saved_model_path, save_format='tf')
        else:
            saved_model_path = self.input_model
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=trt.TrtPrecisionMode.INT8,
            max_workspace_size_bytes=8000000000,
            use_calibration=True)
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=saved_model_path,
            conversion_params=conversion_params)
        converter.convert(calibration_input_fn=self.calibration_input_fn)
        converter.save(output_saved_model_dir=self.target_dir)

    @staticmethod
    def _write_version_file(saved_model_path):
        version_path = os.path.dirname(saved_model_path)
        if isinstance(version_path, bytes):
            version_path = version_path.decode('utf-8')
        with open(os.path.join(str(version_path), "TFVERSION"), 'w', encoding='utf-8') as version_file:
            version_file.write(tf.__version__)
        _LOGGER.info('Write TFVERSION file success, path: %s', version_path)

    @staticmethod
    def get_platform():
        """
        Get platform
        :return:
        """
        return "tensorflow", tf.__version__
