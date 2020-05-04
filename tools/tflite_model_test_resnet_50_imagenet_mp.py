# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from common.model_predict import tflite_model_predict
from mpi4py import MPI

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    base_dir = os.path.dirname(__file__)
    request = {
        "dataset": "imagenet",  # or imagenet
        "model_name": "resnet_50",
        "data_dir": os.path.join(base_dir, "../examples/data/imagenet"),
        "batch_size": 1,
        "batch_size_val": 1
    }
    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()
    tflite_file_path = os.path.join(
        base_dir,
        '../examples/models_eval_ckpt/resnet_50_imagenet_quantized/resnet_50/1/resnet_50.tflite')
    tflite_model_predict(request, tflite_file_path, mpi_size, mpi_rank, comm)


