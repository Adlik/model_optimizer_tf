# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This is a get flops sample with specified model
"""
import os

# If you did not execute the setup.py, uncomment the following four lines
# import sys
# from os.path import abspath, join, dirname
# sys.path.insert(0, join(abspath(dirname(__file__)), '../src'))
# print(sys.path)

from model_optimizer.stat import get_keras_model_params_flops  # noqa: E402


def _main():
    base_dir = os.path.dirname(__file__)
    model_h5_path = './models_eval_ckpt/resnet_50_imagenet/checkpoint-90.h5'
    origin_params, origin_flops = get_keras_model_params_flops(os.path.join(base_dir, model_h5_path))
    model_h5_path = './models_eval_ckpt/resnet_50_imagenet_pruned/checkpoint-120.h5'
    pruned_params, pruned_flops = get_keras_model_params_flops(os.path.join(base_dir, model_h5_path))

    print(f'Before prune, FLOPs: {origin_flops}, Params: {origin_params}')
    print(f'After pruned, FLOPs: {pruned_flops}, Params: {pruned_params}')


if __name__ == "__main__":
    _main()
