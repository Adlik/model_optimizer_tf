# Copyright 2022 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Train a ResNet_34 model on the ImageNet dataset
"""
import os

# import sys
# from os.path import abspath, join, dirname
# sys.path.insert(0, join(abspath(dirname(__file__)), '../src'))
# print(sys.path)

from model_optimizer_tf import prune_model  # noqa: E402


def _main():
    base_dir = os.path.dirname(__file__)
    request = {
        "dataset": "imagenet",
        "model_name": "resnet_34",
        "data_dir": os.path.join(base_dir, "./data/imagenet"),
        "batch_size": 256,
        "batch_size_val": 100,
        "learning_rate": 0.1,
        "epochs": 90,
        "checkpoint_path": os.path.join(base_dir, "./models_ckpt/resnet_34_imagenet"),
        "checkpoint_save_period": 5,  # save a checkpoint every 5 epoch
        "checkpoint_eval_path": os.path.join(base_dir, "./models_eval_ckpt/resnet_34_imagenet"),
        "scheduler": "train"
    }
    prune_model(request)


if __name__ == "__main__":
    _main()
