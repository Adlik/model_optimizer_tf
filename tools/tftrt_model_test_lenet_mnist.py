# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test acc for lenet TF-TRT model
"""
import os
from common.model_predict import tftrt_model_predict  # pylint: disable=import-error,no-name-in-module

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    saved_model_dir = os.path.join(
        base_dir,
        '../examples/models_eval_ckpt/lenet_mnist_quantized/lenet_tftrt/1')
    request = {
        "dataset": "mnist",
        "model_name": "lenet",
        "data_dir": os.path.join(base_dir, "../examples/data/mnist"),
        "batch_size": 1,
        "batch_size_val": 1
    }

    tftrt_model_predict(request, saved_model_dir)
