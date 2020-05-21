# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convert Cifar10 Dataset to local TFRecords
"""

import argparse
import os
import tensorflow as tf
from common.convert_to_tfrecord import convert  # pylint: disable=import-error,no-name-in-module

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir',
        default='./data/cifar10',
        help='Directory where TFRecords will be stored')

    args = parser.parse_args()
    (x_train, y_train), (x_test, y_test) = \
        tf.keras.datasets.cifar10.load_data()
    convert((x_train, y_train), 'train', os.path.expanduser(args.data_dir))
    convert((x_test, y_test), 'test', os.path.expanduser(args.data_dir))
