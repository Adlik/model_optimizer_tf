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

from model_optimizer.stat import get_keras_model_flops  # noqa: E402


def _main():
    base_dir = os.path.dirname(__file__)
    model_h5_path = './models_eval_ckpt/lenet_mnist/checkpoint-12.h5'
    origin_flops = get_keras_model_flops(os.path.join(base_dir, model_h5_path))
    model_h5_path = './models_eval_ckpt/lenet_mnist_pruned/checkpoint-12.h5'
    pruned_flops = get_keras_model_flops(os.path.join(base_dir, model_h5_path))

    print('flops before prune: {}'.format(origin_flops))
    print('flops after pruned: {}'.format(pruned_flops))


if __name__ == "__main__":
    _main()
