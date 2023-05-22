# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This is an example of pruning and ensemble distillation of the resnet50 model
Please download the two models senet154 and resnet152b to the directory configured in the file
resnet_50_imagenet_0.5_distill.yaml.
wget -c -O resnet152b-0431-b41ec90e.tf2.h5.zip https://github.com/osmr/imgclsmob/releases/
download/v0.0.517/resnet152b-0431-b41ec90e.tf2.h5.zip
wget -c -O senet154-0466-f1b79a9b_tf2.h5.zip https://github.com/osmr/imgclsmob/releases/
download/v0.0.422/senet154-0466-f1b79a9b_tf2.h5.zip
"""
import os
# If you did not execute the setup.py, uncomment the following four lines
# import sys
# from os.path import abspath, join, dirname
# sys.path.insert(0, join(abspath(dirname(__file__)), '../src'))
# print(sys.path)

from model_optimizer_tf import prune_model  # noqa: E402


def _main():
    base_dir = os.path.dirname(__file__)
    request = {
        "dataset": "imagenet",
        "model_name": "resnet_50",
        "data_dir": os.path.join(base_dir, "/data/imagenet/tfrecord-dataset"),
        "batch_size": 256,
        "batch_size_val": 100,
        "learning_rate": 0.1,
        "epochs": 360,
        "checkpoint_path": os.path.join(base_dir, "./models_ckpt/resnet_50_imagenet_pruned"),
        "checkpoint_save_period": 5,  # save a checkpoint every 5 epoch
        "checkpoint_eval_path": os.path.join(base_dir, "./models_eval_ckpt/resnet_50_imagenet_pruned"),
        "scheduler": "uniform_auto",
        "is_distill": True,
        "scheduler_file_name": "resnet_50_imagenet_0.5_distill.yaml"
    }
    os.environ['L2_WEIGHT_DECAY'] = "5e-5"
    prune_model(request)


if __name__ == "__main__":
    _main()
