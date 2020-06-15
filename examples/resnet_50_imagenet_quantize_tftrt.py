# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This is a imagenet sample which compile keras h5 model to tf TF-TRT model.
The request of compiling model must match config_schema.json
"""
import os
# If you did not execute the setup.py, uncomment the following four lines
# import sys
# from os.path import abspath, join, dirname
# sys.path.insert(0, join(abspath(dirname(__file__)), '../src'))
# print(sys.path)
from model_optimizer import quantize_model  # noqa: E402
from model_optimizer.quantizer.calib_dataset import input_fn  # noqa: E402


def _main():
    base_dir = os.path.dirname(__file__)
    request = {
        "model_type": "tftrt",  # or tftrt
        "model_name": "resnet_50_tftrt",
        "input_model": os.path.join(base_dir, "./models_eval_ckpt/resnet_50_imagenet_pruned/checkpoint-120.h5"),
        "export_path": os.path.join(base_dir, "./models_eval_ckpt/resnet_50_imagenet_quantized"),
    }
    tfrecord_filename = './data/imagenet_tiny/imagenet_tiny_100.tfrecord'
    result = quantize_model(request, input_fn('imagenet', tfrecord_filename))
    print(result)


if __name__ == "__main__":
    _main()
