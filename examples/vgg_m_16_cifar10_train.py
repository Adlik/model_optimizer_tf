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
        "dataset": "cifar10",
        "model_name": "vgg_m_16",
        "data_dir": os.path.join(base_dir, "./data/cifar10"),
        "batch_size": 128,
        "batch_size_val": 100,
        "learning_rate": 0.1,
        "epochs": 160,
        "checkpoint_path": os.path.join(base_dir, "./models_ckpt/vgg_m_16_cifar10"),
        "checkpoint_save_period": 20,  # save a checkpoint every epoch
        "checkpoint_eval_path": os.path.join(base_dir, "./models_eval_ckpt/vgg_m_16_cifar10"),
        "scheduler": "train"
    }
    prune_model(request)

