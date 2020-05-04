# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from common.model_predict import keras_model_predict


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    request = {
        "dataset": "mnist",
        "model_name": "lenet",
        "data_dir": os.path.join(base_dir, "../examples/data/mnist"),
        "batch_size": 64,
        "batch_size_val": 64
    }
    model_path = os.path.join(base_dir, '../examples/models_ckpt/lenet_mnist_pruned/checkpoint-12.h5')
    keras_model_predict(request, model_path)
