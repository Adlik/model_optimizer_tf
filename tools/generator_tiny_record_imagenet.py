# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import random
import glob
import os

if __name__ == "__main__":
    data_dir = '../examples/data/imagenet'
    file_pattern = os.path.join(data_dir, 'train-*-of-*')
    filenames = glob.glob(file_pattern)
    tfrecord_wrt = tf.io.TFRecordWriter('../examples/data/imagenet_tiny/imagenet_tiny_100.tfrecord')

    res = []
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
        c = 0
        for record in tf.compat.v1.io.tf_record_iterator(filename):
            c += 1
            if c == r:
                tfrecord_wrt.write(record)
                break
    tfrecord_wrt.close()
