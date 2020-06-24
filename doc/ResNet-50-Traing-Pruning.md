# ResNet-50 training and pruning

The following uses ResNet-50 on the ImageNet data set to illustrate how to use the model optimizer to achieve model
training, pruning, and quantization.

## 1 Prepare data

### 1.1 Generate training and test data sets

You may follow the data preparation guide [here](https://github.com/tensorflow/models/tree/v1.13.0/research/inception)
to download the full data set and convert it into TFRecord files. By default, when the script finishes, you will find
1024 training files and 128 validation files in the DATA_DIR. The files will match the patterns train-?????-of-01024
and validation-?????-of-00128, respectively.

### 1.2 Generate small batch data sets required for int-8 quantization

Enter the tools directory and execute

```shell
cd tools
python generator_tiny_record_imagenet.py
```

By default, the imagenet_tiny_100.tfrecord file will be generated in the ../examples/data/imagenet_tiny directory.

### 2 Train

Enter the examples directory and execute

```shell
cd examples
horovodrun -np 8 -H localhost:8 python resnet_50_imagenet_train.py
```

After execution, the default checkpoint file will be generated in ./models_ckpt/resnet_50_imagenet, and the inference
checkpoint file will be generated in ./models_eval_ckpt/resnet_50_imagenet. You can also modify the checkpoint_path and
checkpoint_eval_path of the resnet_50_imagenet_train.py file to change the generated file path.

### 3 Prune

If you have done the complete training, you can copy the checkpoint-90.h5 file in the ./models_ckpt/resnet_50_imagenet
directory to the ./models_ckpt/resnet_50_imagenet_pruned directory and rename it to checkpoint-50.h5, and then perform
pruning.
Enter the examples directory and execute

```shell
cd examples
cp ./models_ckpt/resnet_50_imagenet/checkpoint-90.h5 ./models_ckpt/resnet_50_imagenet_pruned/checkpoint-50.h5
horovodrun -np 8 -H localhost:8 python resnet_50_imagenet_prune_30_bn.py
```

Or you can start a training and pruning process from scratch

```shell
cd examples
horovodrun -np 8 -H localhost:8 python resnet_50_imagenet_prune_30_bn.py
```

After execution, the default checkpoint file will be generated in ./models_ckpt/resnet_50_imagenet_pruned, and the
inference checkpoint file will be generated in ./models_eval_ckpt/resnet_50_imagenet_pruned. You can also modify the
checkpoint_path and checkpoint_eval_path of the resnet_50_imagenet_train.py file to change the generated file path.

### 4 Quantize and generate a TensorFlow Lite FlatBuffer file

Enter the examples directory and execute

```shell
cd examples
python resnet_50_imagenet_quantize_tflite.py
```

After execution, the default checkpoint file will be generated in ./models_ckpt/resnet_50_imagenet_pruned, and the
tflite file will be generated in ./models_eval_ckpt/resnet_50_imagenet_quantized. You can also modify the export_path
of the resnet_50_imagenet_quantize.py file to change the generated file path.

You can enter the tools directory and execute

```shell
cd tools
python tflite_model_test_resnet_50_imagenet.py
```

or

```shell
cd tools
mpirun --allow-run-as-root -np 64 -H localhost:64 python tflite_model_test_resnet_50_imagenet_mp.py
```

Verify accuracy after quantization

### 5 Quantize and generate a TensorFlow with TensorRT (TF-TRT) file

Enter the examples directory and execute

```shell
cd examples
python resnet_50_imagenet_quantize_tftrt.py
```

After execution, the savedmodel file will be generated in ./models_eval_ckpt/resnet_50_imagenet_quantized
/resnet_50_imagenet_tftrt/1 by default. You can also modify the export_path of the resnet_50_imagenet_quantize.py file
to change the generated file path.
You can enter the directory and execute

```shell
cd tools
python tftrt_model_test_resnet_50_imagenet.py
```

Verify accuracy after quantization
