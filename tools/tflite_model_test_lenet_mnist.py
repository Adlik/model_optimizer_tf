# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from common.model_predict import tflite_model_predict


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    base_dir = os.path.dirname(__file__)
    request = {
        "dataset": "mnist",  # or imagenet
        "model_name": "lenet",
        "data_dir": os.path.join(base_dir, "../examples/data/mnist"),
        "batch_size": 1,
        "batch_size_val": 1
    }
    tflite_file_path = os.path.join(base_dir, '../examples/models_eval_ckpt/lenet_mnist_quantized/lenet/1/lenet.tflite')
    tflite_model_predict(request, tflite_file_path)
