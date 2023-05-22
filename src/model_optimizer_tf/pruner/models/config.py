# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Get the configuration, if the environment variable exists, get it from the environment variable.
If it does not exist, get it from the initial value.
"""
import os


class ModelConfig:
    """
    Model train config parameters
    """
    def __init__(self, l2_weight_decay=1e-4, batch_norm_decay=0.95, batch_norm_epsilon=0.001, std_dev=0.09):
        self._l2_weight_decay = l2_weight_decay
        self._std_dev = std_dev
        self._batch_norm_decay = batch_norm_decay
        self._batch_norm_epsilon = batch_norm_epsilon

    @property
    def l2_weight_decay(self):
        """
        l2_weight_decay
        :return: l2_weight_decay
        """
        if 'L2_WEIGHT_DECAY' in os.environ:
            return float(os.environ['L2_WEIGHT_DECAY'])
        else:
            return self._l2_weight_decay

    @property
    def batch_norm_decay(self):
        """
        batch_norm_decay
        :return: batch_norm_decay
        """
        if 'BATCH_NORM_DECAY' in os.environ:
            return float(os.environ['BATCH_NORM_DECAY'])
        else:
            return self._batch_norm_decay

    @property
    def batch_norm_epsilon(self):
        """
        batch_norm_epsilon
        :return: batch_norm_epsilon
        """
        if 'BATCH_NORM_EPSILON' in os.environ:
            return float(os.environ['BATCH_NORM_EPSILON'])
        else:
            return self._batch_norm_epsilon

    @property
    def std_dev(self):
        """
        std_dev
        :return: std_dev
        """
        if 'STD_DEV' in os.environ:
            return float(os.environ['STD_DEV'])
        else:
            return self._std_dev
