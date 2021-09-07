# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Train a cnn1d model on iscx_session_all dataset
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
        "dataset": "iscx_session_all",
        "model_name": "cnn1d",
        "data_dir": "/data/12class/SessionAllLayers",
        "batch_size": 500,
        "batch_size_val": 100,
        "learning_rate": 1e-3,
        "epochs": 60,
        "checkpoint_path": os.path.join(base_dir, "./models_ckpt/cnn1d"),
        "checkpoint_save_period": 1,  # save a checkpoint every epoch
        "checkpoint_eval_path": os.path.join(base_dir, "./models_eval_ckpt/cnn1d"),
        "scheduler": "train"
    }
    prune_model(request)


if __name__ == "__main__":
    _main()
