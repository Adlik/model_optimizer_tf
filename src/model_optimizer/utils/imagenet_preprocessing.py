# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Provides utilities to preprocess images
"""

import tensorflow as tf

_RESIZE_MIN = 256


def preprocess_image(image_buffer, bbox, output_height, output_width,
                     num_channels=3, is_training=False):
    """
    Image process
    :param image_buffer: image
    :param bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    :param output_height: output height
    :param output_width: output width
    :param num_channels: num of channels
    :param is_training: if training or not
    :return: image
    """
    if is_training:
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            tf.image.extract_jpeg_shape(image_buffer),
            bounding_boxes=bbox,
            min_object_covered=0.1,
            aspect_ratio_range=[0.75, 1.33],
            area_range=[0.05, 1.0],
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)

        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

        cropped = tf.image.decode_and_crop_jpeg(
            image_buffer, crop_window, channels=num_channels)

        cropped = tf.image.random_flip_left_right(cropped)
        return tf.image.resize(cropped, [output_height, output_width], method=tf.image.ResizeMethod.BILINEAR)
    else:
        image = tf.image.decode_jpeg(image_buffer, channels=num_channels)

        # Resize images preserving the original aspect ratio
        shape = tf.shape(input=image)
        height, width = shape[0], shape[1]
        resize_min = tf.cast(_RESIZE_MIN, tf.float32)
        height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
        smaller_dim = tf.minimum(height, width)
        scale_ratio = resize_min / smaller_dim
        new_height = tf.cast(height * scale_ratio, tf.int32)
        new_width = tf.cast(width * scale_ratio, tf.int32)
        image = tf.image.resize(image, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)

        # Performs central crops of the given image
        shape = tf.shape(input=image)
        height, width = shape[0], shape[1]
        crop_top = (height - output_height) // 2
        crop_left = (width - output_width) // 2
        return tf.slice(image, [crop_top, crop_left, 0], [output_height, output_width, -1])
