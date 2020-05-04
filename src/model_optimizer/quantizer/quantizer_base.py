# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Model quantizer base class
"""
from abc import abstractmethod
from .compressor import compress_dir
import shutil
import uuid
import os
from .message import fail, success
from ..log_util import get_logger
_LOGGER = get_logger(__name__)


class BaseQuantizer:
    """
    Quantizer base class
    """
    _COMMON_PARAMS = [
        "input_model",
        "model_name",
        "export_path",
        "version"
    ]
    _COMMON_REQUIRED = [
        "input_model",
        "model_name",
        "export_path"
    ]

    def __init__(self, config):
        for item in self._COMMON_PARAMS:
            if config.get_attribute(item) is None and item in self._COMMON_REQUIRED:
                _LOGGER.error('Require "%s" but not found', item)
                raise Exception('Require "%s" but not found' % item)
            self.__setattr__(item, config.get_attribute(item))
        self.model_dir = self._make_model_dir()
        self.version, self.version_dir = self._get_version_dir()
        self.target_dir = self._make_target_dir()
        self.custom_object = None
        _LOGGER.info('Output dir is: %s, version: %s', self.model_dir, self.version)

    def quantize(self):
        """
        Quantize model
        :return: Return quantize result
        """
        try:
            self._do_quantize()
            os.rename(self.target_dir, self.version_dir)
            zip_path = self._compress([self.version_dir])
            return success(zip_path)
        except Exception as error:  # pylint:disable=broad-except
            _LOGGER.error('Quantize model failed, error: %s', error)
            _LOGGER.exception(error)
            self._cleanup()
            return fail(str(error))

    def _compress(self, source_list):
        """
        Compress model to .zip
        :return:
        """
        # self.target_dir -> modelName_version.zip
        zip_file_path = os.path.join(self.export_path, self.model_name + '_' + str(self.version) + '.zip')
        return compress_dir(source_list, zip_file_path)

    def _make_model_dir(self):
        """
        Make model dir, the structure of export dir is:
        export_dir
        └── model_name
            ├── version_1(version_dir)
            │   └── tftrt SavedModel or tflite model
            └── version_2
                └── tftrt SavedModel or tflite model
        :return:
        """
        _LOGGER.info('make_model_dir: export base path: %s', self.export_path)
        if not os.path.exists(self.export_path):
            os.makedirs(self.export_path, exist_ok=True)
        model_dir = os.path.join(self.export_path, self.model_name)
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    def _get_version_dir(self):
        version = getattr(self, "version", None)
        if version is None:
            version = self._get_model_default_version()
        version = str(version)
        version_dir = os.path.join(self.model_dir, version)
        _LOGGER.info("Export model version : %s, dir: %s", version, version_dir)
        if os.path.exists(version_dir):
            raise Exception('Output version is already exist: {}'.format(version_dir))
        return version, version_dir

    def _get_model_default_version(self):
        sub_dirs = [int(child) for child in os.listdir(self.model_dir)
                    if os.path.isdir(os.path.join(self.model_dir, child)) and child.isdigit()]
        sub_dirs.sort()
        version = str(sub_dirs[-1] + 1) if sub_dirs else "1"
        return version

    def _make_target_dir(self):
        temp_dir_name = str(uuid.uuid3(uuid.NAMESPACE_URL, '_'.join([self.model_name, self.version])))
        _LOGGER.info("temporary export dir: %s, %s", temp_dir_name, os.path.join(self.model_dir, temp_dir_name))
        target_dir = os.path.join(self.model_dir, temp_dir_name)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        return target_dir

    def _cleanup(self):
        if os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir)
        if os.path.exists(self.version_dir):
            shutil.rmtree(self.version_dir)

    @abstractmethod
    def _do_quantize(self):
        pass
