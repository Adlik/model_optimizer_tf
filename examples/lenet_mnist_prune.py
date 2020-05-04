# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

# If you have executed the setup.py, comment out the following three lines
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '../src'))
print(sys.path)

from model_optimizer import prune_model # noqa: E402
if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    request = {
        "dataset": "mnist",  # or imagenet
        "model_name": "lenet",
        "data_dir": os.path.join(base_dir, "./data/mnist"),
        "batch_size": 120,
        "batch_size_val": 100,
        "learning_rate": 0.001,
        "epochs": 12,
        "checkpoint_path": os.path.join(base_dir, "./models_ckpt/lenet_mnist_pruned"),
        "checkpoint_save_period": 1,   # save a checkpoint every n epoch
        "checkpoint_eval_path": os.path.join(base_dir, "./models_eval_ckpt/lenet_mnist_pruned"),
        "scheduler": "uniform_auto"
    }
    prune_model(request)