# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test acc for resnet-50 TF-TRT model
"""
import os
from common.model_predict import tftrt_model_predict  # pylint: disable=import-error,no-name-in-module

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    saved_model_dir = os.path.join(
        base_dir,
        '../examples/models_eval_ckpt/resnet_50_imagenet_quantized/resnet_50_tftrt/1')
    request = {
        "dataset": "imagenet",
        "model_name": "resnet_50",
        "data_dir": os.path.join(base_dir, "../examples/data/imagenet"),
        "batch_size": 1,
        "batch_size_val": 1
    }

    tftrt_model_predict(request, saved_model_dir)
