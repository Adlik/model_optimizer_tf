# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import random

if __name__ == "__main__":
    data_dir = '../examples/data/mnist'
    filename = data_dir + '/train.tfrecords'
    tfrecord_wrt = tf.io.TFRecordWriter('../examples/data/mnist_tiny/mnist_tiny_100.tfrecord')
    res = []
    while len(res) < 100:
        r = random.randint(1, 60000)
        if r not in res:
            res.append(r)
    c = 0
    for record in tf.compat.v1.io.tf_record_iterator(filename):
        c += 1
        if c in res:
            tfrecord_wrt.write(record)
    tfrecord_wrt.close()
