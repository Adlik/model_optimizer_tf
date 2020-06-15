# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate mnist tf-record with 100 images
"""
import random
import typing
import tensorflow as tf


if __name__ == "__main__":
    DATA_DIR = '../examples/data/mnist'
    FILENAME = DATA_DIR + '/train.tfrecords'
    tfrecord_wrt = tf.io.TFRecordWriter('../examples/data/mnist_tiny/mnist_tiny_100.tfrecord')
    res = []  # type: typing.List[int]
    while len(res) < 100:
        r = random.randint(1, 60000)
        if r not in res:
            res.append(r)
    i = 0
    for record in tf.compat.v1.io.tf_record_iterator(FILENAME):
        i += 1
        if i in res:
            tfrecord_wrt.write(record)
    tfrecord_wrt.close()
