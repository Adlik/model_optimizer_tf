# MobileNet v1 training and pruning

The following uses MobileNet v1 on the ImageNet dataset to illustrate how to use the model optimizer to achieve  
model traing, and pruning

## 1 Prepare data

### 1.1 Generate training and test data sets

You may follow the data preparation guide [here](https://github.com/tensorflow/models/tree/v1.13.0/research/inception)  
to download the full dataset and convert it into TFRecord files. By default, when the script finishes, you will find  
1024 training files and 128 validation files in the DATA_DIR. The file will match the patterns  
train-?????-of-01024 and validation-?????-of-0128, respectively.

## 2 Train

Enter the examples directory and execute

```shell
cd examples
horovodrun -np 8 -H localhost:8 python movilenet_v1_imagenet_train.py
```

After execution, the default checkpoint file will be generated in ./models_ckpt/mobilenet_v1_imagenet, and the  
inference checkpoint file will be generated in ./models_eval_ckpt/mobilenet_v1_imagenet. You can also modify the  
checkpoint_path and checkpoint_eval_path of the mobilenet_v1_imagenet_train.py file to change the generated file path.

## 3 Prune

Here, you can use a full trained model or the  model in training process as a initial model to prune.  The following  
uses specified pruning strategy as an example.

If you have a well trained model, for example, named checkpoint-120.h5 in directory ./models_ckpt/mobilenet_v1_imagenet.
You can copy it to the ./models_ckpt/mobilenet_v1_imagenet_specified_pruned directory, and then perform pruning. Enter  
the examples diretory and execute

```shell
cd examples
cp ./models_ckpt/mobilenet_v1_imagenet/checkpoint-120.h5 ./models_ckpt/mobilenet_v1_imagenet_specified_pruned/
horovodrun -np 8 -H localhost:8 python mobilenet_v1_imagenet_prune.py
```

Or you can start a training and pruning process from scratch

```shell
cd examples
horovodrun -np 8 -H localhost:8 python mobilenet_v1_imagenet_prune.py
```

After execution, the default checkpoint file weill be generated in ./models_ckpt/mobilenet_v1_imagenet_specified_pruned,
and the inference checkpoint file will be generated in ./models_eval_ckpt/mobilenet_v1_imagenet_specified_pruned. You  
can also modify the checkpoint_path and checkpoint_eval_path of the mobilenet_v1_imagenet_prune.py file to change the  
generated file path.

## 4 Quantize

To be continue.
