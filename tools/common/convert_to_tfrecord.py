# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convert to tf-record
"""
import os
import tensorflow as tf


def _int64_feature(value: int) -> tf.train.Features.FeatureEntry:
    """
    Create a Int64List Feature
    :param value: The value to store in the feature
    :return: The FeatureEntry
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value: str) -> tf.train.Features.FeatureEntry:
    """
    Create a BytesList Feature
    :param value: The value to store in the feature
    :return: The FeatureEntry
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _data_path(data_dir: str, name: str) -> str:
    """
    Construct a full path to a TFRecord file to be stored in the
    data_directory.
    :param data_dir: directory where the records will be stored
    :param name: name of the TFRecord
    :return: full path to the TFRecord file
    """
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    return os.path.join(data_dir, f'{name}.tfrecords')


def _process_examples(images, labels, start_idx: int, end_idx: int, file_name: str):
    """
    Convert the dataset into TFRecords on disk
    :param images:
    :param labels:
    :param start_idx: The start index of record
    :param end_idx: The end index of record
    :param file_name:
    :return:
    """
    num_examples = end_idx - start_idx
    with tf.io.TFRecordWriter(file_name) as writer:
        for index in range(start_idx, end_idx):
            print(f"\rProcessing sample {index + 1} of {num_examples}", end='', flush=True)
            image_raw = images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _bytes_feature(image_raw),
                'label': _int64_feature(int(labels[index]))
            }))
            writer.write(example.SerializeToString())


def convert(data_set, name: str, data_dir: str, num_shards: int = 1):
    """
    Convert the dataset into TFRecords on disk
    :param data_set: The data set to convert
    :param name: The name of the data set
    :param data_dir: The directory where records will be stored
    :param num_shards: The number of files on disk to separate records into
    :return:
    """
    print(f'\nProcessing {name} data')
    images, labels = data_set
    num_examples = images.shape[0]
    if num_shards == 1:
        _process_examples(images, labels, 0, num_examples, _data_path(data_dir, name))
    else:
        total_examples = num_examples
        samples_per_shard = total_examples // num_shards
        for shard in range(num_shards):
            start_index = shard * samples_per_shard
            end_index = start_index + samples_per_shard
            _process_examples(images, labels, start_index, end_index,
                              _data_path(data_dir, f'{name}-{shard + 1}-of-{num_shards}'))
