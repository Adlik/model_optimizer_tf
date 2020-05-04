# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tensorflow as tf


def get_keras_model_flops(model_h5_path):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            _ = tf.keras.models.load_model(model_h5_path)
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)
    tf.compat.v1.reset_default_graph()
    return flops.total_float_ops


def print_keras_model_summary(model, hvd_rank):
    if hvd_rank != 0:
        return
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    short_model_summary = "\n".join(string_list)
    print(short_model_summary)
