# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
_Config Object
"""
import os
import json
import jsonschema
from ..log_util import get_logger

_LOGGER = get_logger(__name__)


class _Config:
    def __init__(self, **kwargs):
        self._config = kwargs
        _LOGGER.info('Dump config: %s', self._config)

    def get_attribute(self, name, default=None):
        """
        Get attribute by name
        :param name:
        :param default:
        :return:
        """
        return self._config.get(name, default)


def create_config_from_obj(obj) -> object:
    """
    Create message config from a dictionary which must match config_schema
    :param obj: dict
    :return:
    """
    schema_path = os.path.join(os.path.dirname(__file__), 'config_schema.json')
    with open(schema_path) as schema_file:
        body_schema = json.load(schema_file)

    jsonschema.validate(obj, body_schema)
    return _Config(**obj)
