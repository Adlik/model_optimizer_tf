# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Quantize or prune model
"""
import os
from .quantizer import create_quantizer
from .quantizer.config import create_config_from_obj as quant_conf_from_obj
from .pruner.config import create_config_from_obj as prune_conf_from_obj
from .pruner.runner import run_scheduler


def _make_dirs(file_path):
    os.makedirs(file_path, exist_ok=True)


def quantize_model(request, calibration_input_fn):
    """
    Quantize serving model
    :param request: dict, must match quantizer config_schema.json
    :param calibration_input_fn: a generator function that yields input data as a
        list or tuple, which will be used to execute the converted signature for
        calibration.
    :return:
    """
    _make_dirs(request['export_path'])
    optimizer = create_quantizer(quant_conf_from_obj(request), calibration_input_fn)
    return optimizer.quantize()


def prune_model(request):
    """
    Prune model
    :param request: dict, must match pruner config_schema.json
    :return:
    """
    _make_dirs(request['checkpoint_path'])
    _make_dirs(request['checkpoint_eval_path'])
    return run_scheduler(prune_conf_from_obj(request))
