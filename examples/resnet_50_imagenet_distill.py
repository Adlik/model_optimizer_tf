# Copyright 2020 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Distill a ResNet_50 model from a trained ResNet_101 model on the ImageNet dataset
"""
import os
# If you did not execute the setup.py, uncomment the following four lines
# import sys
# from os.path import abspath, join, dirname
# sys.path.insert(0, join(abspath(dirname(__file__)), '../src'))
# print(sys.path)

from model_optimizer import prune_model  # noqa: E402


def _main():
    base_dir = os.path.dirname(__file__)
    request = {
        "dataset": "imagenet",
        "model_name": "resnet_50",
        "data_dir": os.path.join(base_dir, "./data/imagenet"),
        "batch_size": 256,
        "batch_size_val": 100,
        "learning_rate": 0.1,
        "epochs": 90,
        "checkpoint_path": os.path.join(base_dir, "./models_ckpt/resnet_50_imagenet_distill"),
        "checkpoint_save_period": 5,  # save a checkpoint every 5 epoch
        "checkpoint_eval_path": os.path.join(base_dir, "./models_eval_ckpt/resnet_50_imagenet_distill"),
        "scheduler": "distill",
        "scheduler_file_name": "resnet_50_imagenet_0.3.yaml",
        "classifier_activation": None  # None or "softmax", default is softmax
    }
    prune_model(request)


if __name__ == "__main__":
    _main()
