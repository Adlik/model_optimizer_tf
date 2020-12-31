# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Imagenet dataset.
"""
import os
import tensorflow as tf
from .dataset_base import DatasetBase
from ...utils.imagenet_preprocessing import preprocess_image


class ImagenetDataset(DatasetBase):
    """
    Imagenet dataset.
    """
    def __init__(self, config, is_training, num_shards=1, shard_index=0):
        """
        Constructor function.
        :param config: Config object
        :param is_training: whether to construct the training subset
        :return:
        """
        super(ImagenetDataset, self).__init__(config, is_training, num_shards, shard_index)
        if is_training:
            self.file_pattern = os.path.join(self.data_dir, 'train-*-of-*')
            self.batch_size = self.batch_size
        else:
            self.file_pattern = os.path.join(self.data_dir, 'validation-*-of-*')
            self.batch_size = self.batch_size_val
        self.dataset_fn = tf.data.TFRecordDataset
        self.buffer_size = 10000
        self.num_samples_of_train = 1281167
        self.num_samples_of_val = 50000


    def parse_fn(self, example_serialized):
        """
        Parse features from the serialized data
        :param example_serialized: serialized data
        :return: image,label
        """
        feature_description = {
            'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
            'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string, default_value='')
        }
        sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
        # Sparse features in Example proto.
        feature_description.update(
            {k: sparse_float32 for k in [
                'image/object/bbox/xmin', 'image/object/bbox/ymin',
                'image/object/bbox/xmax', 'image/object/bbox/ymax']})

        features = tf.io.parse_single_example(serialized=example_serialized,
                                              features=feature_description)
        label = tf.cast(features['image/class/label'], dtype=tf.int32)

        xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
        ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
        xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
        ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

        # Note that we impose an ordering of (y, x) just to make life difficult.
        bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

        # Force the variable number of bounding boxes into the shape
        # [1, num_boxes, coords].
        bbox = tf.expand_dims(bbox, 0)
        bbox = tf.transpose(a=bbox, perm=[0, 2, 1])

        image_buffer = features['image/encoded']
        image = preprocess_image(
            image_buffer=image_buffer,
            bbox=bbox,
            output_height=224,
            output_width=224,
            num_channels=3,
            is_training=self.is_training)
        return image, label

    def parse_fn_distill(self, example_serialized):
        """
        Parse features from the serialized data for distillation
        :param example_serialized: serialized data
        :return: {image, label},{}
        """
        feature_description = {
            'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
            'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string, default_value='')
        }
        sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
        # Sparse features in Example proto.
        feature_description.update(
            {k: sparse_float32 for k in [
                'image/object/bbox/xmin', 'image/object/bbox/ymin',
                'image/object/bbox/xmax', 'image/object/bbox/ymax']})

        features = tf.io.parse_single_example(serialized=example_serialized,
                                              features=feature_description)
        label = tf.cast(features['image/class/label'], dtype=tf.int32)

        xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
        ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
        xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
        ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

        # Note that we impose an ordering of (y, x) just to make life difficult.
        bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

        # Force the variable number of bounding boxes into the shape
        # [1, num_boxes, coords].
        bbox = tf.expand_dims(bbox, 0)
        bbox = tf.transpose(a=bbox, perm=[0, 2, 1])

        image_buffer = features['image/encoded']
        image = preprocess_image(
            image_buffer=image_buffer,
            bbox=bbox,
            output_height=224,
            output_width=224,
            num_channels=3,
            is_training=self.is_training)
        inputs = {"image": image, "label": label}
        targets = {}
        return inputs, targets
