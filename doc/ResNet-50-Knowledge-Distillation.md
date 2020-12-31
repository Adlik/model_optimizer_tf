# ResNet-50 Knowledge Distillation

The following uses ResNet-101 on the ImageNet data set as teacher model to illustrate how to use the model optimizer to
improve the preformance of ResNet-50 by knowledge distillation.

## 1 Prepare data

### 1.1 Generate training and test data sets

You may follow the data preparation guide [here](https://github.com/tensorflow/models/tree/v1.13.0/research/inception)
to download the full data set and convert it into TFRecord files. By default, when the script finishes, you will find
1024 training files and 128 validation files in the DATA_DIR. The files will match the patterns train-?????-of-01024
and validation-?????-of-00128, respectively.

### 2 Train the teacher model

Enter the examples directory and execute

```shell
cd examples
horovodrun -np 8 -H localhost:8 python resnet_101_imagenet_train.py
```

After execution, the default checkpoint file will be generated in ./models_ckpt/resnet_101_imagenet, and the inference
checkpoint file will be generated in ./models_eval_ckpt/resnet_101_imagenet. You can also modify the checkpoint_path
and checkpoint_eval_path of the resnet_101_imagenet_train.py file to change the generated file path.

### 3 Distill

Enter the examples directory and execute

```shell
horovodrun -np 8 -H localhost:8 python resnet_50_imagenet_distill.py
```
After execution, the default checkpoint file will be generated in ./models_ckpt/resnet_50_imagenet_distill, and
the inference checkpoint file will be generated in ./models_eval_ckpt/resnet_50_imagenet_distill. You can also 
modify the checkpoint_path and checkpoint_eval_path of the resnet_50_imagenet_distill.py file to change the generated 
file path.

> Note
>
> > i. The model in the checkpoint_path is not the pure ResNet-50 model. It's the hybird of ResNet-50(student) and 
> > ResNet-101(teacher)
> >
> > ii. The model in the checkpoint_eval_path is the distilled model, i.e. pure ResNet-50 model




