# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys


# If you have executed the setup.py, comment out the following three lines
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '../src'))
print(sys.path)

from model_optimizer.stat import get_keras_model_flops # noqa: E402


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    model_h5_path = './models_eval_ckpt/resnet_50_imagenet/checkpoint-90.h5'
    origin_flops = get_keras_model_flops(os.path.join(base_dir, model_h5_path))
    model_h5_path = './models_eval_ckpt/resnet_50_imagenet_pruned/checkpoint-120.h5'
    pruned_flops = get_keras_model_flops(os.path.join(base_dir, model_h5_path))

    print('flops before prune: {}'.format(origin_flops))
    print('flops after pruned: {}'.format(pruned_flops))