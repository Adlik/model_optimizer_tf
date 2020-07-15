# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Train a Lenet model on the Mnist dataset
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
        "dataset": "mnist",
        "model_name": "lenet",
        "data_dir": os.path.join(base_dir, "./data/mnist"),
        "batch_size": 120,
        "batch_size_val": 100,
        "learning_rate": 0.001,
        "epochs": 12,
        "checkpoint_path": os.path.join(base_dir, "./models_ckpt/lenet_mnist"),
        "checkpoint_save_period": 1,  # save a checkpoint every epoch
        "checkpoint_eval_path": os.path.join(base_dir, "./models_eval_ckpt/lenet_mnist"),
        "scheduler": "train"
    }
    prune_model(request)


if __name__ == "__main__":
    _main()
