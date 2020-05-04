# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from common.model_predict import keras_model_predict


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    request = {
        "dataset": "imagenet",
        "model_name": "resnet_50",
        "data_dir": os.path.join(base_dir, "../examples/data/imagenet"),
        "batch_size": 64,
        "batch_size_val": 64
    }
    model_path = os.path.join(base_dir, '../examples/models_eval_ckpt/resnet_50_imagenet_pruned/checkpoint-120.h5')
    keras_model_predict(request, model_path)
