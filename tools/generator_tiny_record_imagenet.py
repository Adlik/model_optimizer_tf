# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate imagenet tf-record with 100 images
"""
import os
import glob
import random
import typing
import tensorflow as tf


if __name__ == "__main__":
    DATA_DIR = '../examples/data/imagenet'
    file_pattern = os.path.join(DATA_DIR, 'train-*-of-*')
    filenames = glob.glob(file_pattern)
    tfrecord_wrt = tf.io.TFRecordWriter('../examples/data/imagenet_tiny/imagenet_tiny_100.tfrecord')

    res = []  # type: typing.List[int]
    r = random.randint(1, 1024)
    while len(res) < 100:
        r = random.randint(1, 1024)
        if r not in res:
            res.append(r)
    i = 0
    for filename in filenames:
        i += 1
        if i not in res:
            continue
        r = random.randint(1, 1250)
        j = 0
        for record in tf.compat.v1.io.tf_record_iterator(filename):
            j += 1
            if j == r:
                tfrecord_wrt.write(record)
                break
    tfrecord_wrt.close()
