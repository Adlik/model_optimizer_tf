# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Model predict
"""
import time
import tensorflow as tf
import numpy as np
from tensorflow.compat.v1.saved_model import signature_constants  # pylint: disable=import-error,no-name-in-module
from tensorflow.compat.v1.saved_model import tag_constants  # pylint: disable=import-error,no-name-in-module
from tensorflow.python.framework import convert_to_constants  # pylint: disable=import-error,no-name-in-module


# If you did not execute the setup.py, uncomment the following four lines
# import sys
# from os.path import abspath, join, dirname
# sys.path.insert(0, join(abspath(dirname(__file__)), '../../src'))
# print(sys.path)

from model_optimizer.pruner.dataset import get_dataset  # noqa: E402
from model_optimizer.pruner.config import create_config_from_obj as prune_conf_from_obj  # noqa: E402


def _get_graph_func(saved_model_dir):
    saved_model_loaded = tf.saved_model.load(
        saved_model_dir, tags=[tag_constants.SERVING])
    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    graph_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)
    return graph_func


def _get_from_saved_model(graph_func, input_data, print_result=False):
    output_data = graph_func(input_data)[0].numpy()
    if print_result:
        print(output_data)
    return output_data


def keras_model_predict(request, file_path, is_multi_output=False):
    """
    Keras model predict
    :param request: dict, must match pruner config_schema.json
    :param file_path: file path
    :param is_multi_output: the flag of multiple output of the model
    :return:
    """
    ds_val = get_dataset(prune_conf_from_obj(request), is_training=False)
    num_samples = ds_val.num_samples
    val_dataset = ds_val.build()
    keras_model = tf.keras.models.load_model(file_path)
    score = 0
    cur_steps = 0
    start = time.time()
    for x_test, y_test in val_dataset:
        if is_multi_output:
            result, _ = keras_model.predict(x_test)
        else:
            result = keras_model.predict(x_test)
        output_data = tf.keras.backend.argmax(result)
        for j in range(y_test.shape[0]):
            if int(output_data[j]) == int(y_test[j]):
                score += 1
        cur_steps += y_test.shape[0]
        acc = 100.0 * score / cur_steps
        print(f'\rcur_steps: {cur_steps}/{num_samples}, acc: {acc}', end='', flush=True)
    print('\r')
    print('=' * 50)
    print(f'inference comsume time: {time.time() - start} s')
    print(f'acc1: {100.0 * score / num_samples}')


def tflite_model_predict(request, file_path, mpi_size=1, mpi_rank=0, comm=None):
    """
    tflite model predict, support multi process
    :param request: dict, must match pruner config_schema.json
    :param file_path: file path
    :param mpi_size: mpi size
    :param mpi_rank: mpi rank
    :param comm: mpi4py.MPI.COMM_WORLD object
    :return:
    """
    ds_val = get_dataset(prune_conf_from_obj(request), is_training=False,
                         num_shards=mpi_size, shard_index=mpi_rank)
    total_num_samples = ds_val.num_samples
    num_samples = total_num_samples // mpi_size
    val_dataset = ds_val.build()
    interpreter = tf.lite.Interpreter(file_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    score = 0
    cur_steps = 0
    start = time.time()
    for x_test, y_test in val_dataset:
        interpreter.set_tensor(input_details[0]['index'], x_test)
        interpreter.invoke()
        output_data = tf.keras.backend.argmax(interpreter.get_tensor(output_details[0]['index']))
        for j in range(y_test.shape[0]):
            if int(output_data[j]) == int(y_test[j]):
                score += 1
        cur_steps += y_test.shape[0]
        acc = 100.0 * score / cur_steps
        if mpi_rank == 0:
            print(f'\rcur_steps: {cur_steps}/{num_samples}, acc: {acc}', end='', flush=True)
    if mpi_size != 1:
        score = comm.gather(score, root=0)
        score = np.sum(np.array(score, dtype=np.float32))
    if mpi_rank == 0:
        print('\r')
        print('=' * 50)
        print(f'inference comsume time: {time.time() - start} s')
        print(f'acc1: {100.0 * score / total_num_samples}')


def tftrt_model_predict(request, file_path):
    """
    TF-TRT model predict
    :param request: dict, must match pruner config_schema.json
    :param file_path: file path
    :return:
    """
    graph_func = _get_graph_func(file_path)
    ds_val = get_dataset(prune_conf_from_obj(request), is_training=False)
    num_samples = ds_val.num_samples
    val_dataset = ds_val.build()
    start = time.time()
    cur_steps = 0
    score = 0
    for x_test, y_test in val_dataset:
        input_data = tf.convert_to_tensor(x_test)
        _out = _get_from_saved_model(graph_func, input_data)
        _output = tf.keras.backend.argmax(_out)
        for j in range(y_test.shape[0]):
            if int(_output[j]) == int(y_test[j]):
                score += 1
        cur_steps += y_test.shape[0]
        acc = 100.0 * score / cur_steps
        print(f'\rcur_steps: {cur_steps}/{num_samples}, acc: {acc}', end='', flush=True)
    print('\r')
    print('=' * 50)
    print(f'inference comsume time: {time.time() - start} s')
    print(f'acc1: {100.0 * score / num_samples}')
