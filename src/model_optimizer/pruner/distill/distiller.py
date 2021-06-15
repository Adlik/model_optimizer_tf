# Copyright 2021 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
get distiller model
"""
import tensorflow as tf

from .distill_loss import DistillLossLayer


def get_distiller(student_model, scheduler_config, teacher_model_load_func=None):
    """
    Get distiller model
    :param student_model: student model function
    :param scheduler_config: scheduler config object
    :param teacher_model_load_func: func to load teacher model
    :return: keras model of distiller
    """

    if teacher_model_load_func is None:
        if "model_load_func" in scheduler_config['distill']:
            teacher_model_load_func = scheduler_config['distill']["model_load_func"]

    input_img = tf.keras.layers.Input(shape=(224, 224, 3), name='image')
    input_lbl = tf.keras.layers.Input((), name="label", dtype='int32')
    student = student_model
    logits = student(input_img)
    total_loss = DistillLossLayer(scheduler_config['distill']['teacher_path'],
                                  scheduler_config['distill']['alpha'],
                                  scheduler_config['distill']['temperature'],
                                  teacher_model_load_func=teacher_model_load_func)([input_img, input_lbl, logits])
    distill_model = tf.keras.Model(inputs=[input_img, input_lbl], outputs=[logits, total_loss])

    return distill_model
