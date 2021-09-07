# Tiny 1D-CNN Knowledge Distillation

The following uses 1D-CNN on the 12 classes session all dataset as teacher model to illustrate how to use the model 
optimizer to improve the preformance of tiny 1D-CNN by knowledge distillation.

The 1D-CNN model is from Wang's paper[Wang, W.; Zhu, M.; Wang, J.; Zeng, X.; Yang, Z. End-to-end encrypted traffic 
classification with one-dimensional convolution neural networks.] The tiny 1D-CNN model is a slim version of the 1D-CNN 
model mentioned before. Using 1D-CNN model as the teacher to ditstill tiny 1D-CNN model, performance can be improved by
5.66%.

The details are shown in the table below, and the code can refer to examples\cnn1d_tiny_iscx_session_all_distill.py. 

| Model     | Accuracy | Params              | Model Size |
| --------- | -------- | -------------------- | ---------------------------- |
| cnn1d | 92.67%   | 5832588 | 23M|
| cnn1d_tiny | 87.62%   | 134988 | 546K|
| cnn1d_tiny+ distill | 93.28%   | 134988 | 546K|


## 1 Create custom dataset
Using [ISCX dataset](https://www.unb.ca/cic/datasets/vpn.html), you can get the processed 12-classes-session-all dataset
from [wang's github](https://github.com/echowei/DeepTraffic/blob/master/2.encrypted_traffic_classification/3.PerprocessResults/12class.zip).
We name the dataset as iscx_session_all. In the iscx_session_all, there are 35501 training samples, the shape is (35501, 28, 28), 
3945 testing samples. 

Now that you have the dataset, you can implement your custom dateset by extending model_optimizer.prunner.dataset.
dataset_base.DatasetBase and implementing:

1. \__init__, required, where you can do all dataset initialization 
2. parse_fn, required, where is  the map function of the dataset 
3. parse_fn_distill, required, where is the map function of the dataset used in distillation
4. build, optional, where is the process of building the dataset. If your dataset is not in tfrecord format, you must 
implement this function.

Here in the custom dataset, we reshape the samples from (None, 28, 28, 1) to (None, 1, 784, 1) for the following 1D-CNN
models.

After that, all you need is put the dataset name in the following files:
1. src/model_optimizer/prunner/config_schema.json the "enum" list
2. src/model_optimizer/prunner/dataset/\__init__.py. Add the dataset name in Line 19 and add the dataset instance in the
if-else clause.

## Create custom model
Create your own model using The Keras functional API in model_optimizer.prunner.models.

After that, all you need is put the model name and initialize the model in the following files:
1. src/model_optimizer/prunner/models/\__init__.py. Add the model name in Line 21 and add the model instance in the
if-else clause.

## Create custom learner
Implement your own learner by extending model_optimizer.prunner.learner.learner_base.LearnerBase and implementing:
1. \__init__, required, where you can define your own learning rate callback
2. get_optimizer, required, where you can define your own optimizer
3. get_losses, required, where you can define your own loss function
4. get_metrics, required, where you can define your own metrics

After that, all you need is put the model name and dataset name and initialize the learner in the following files:
1. src/model_optimizer/prunner/learner/\__init__.py 

## Create the training process of the teacher model, and train the teacher model
Enter the examples directory, create cnn1d_iscx_session_all_train.py for cnn1d model. 

> Note
>  
> > the "model_name" and "dataset" in the request must be the same as you defined before

Execute:

```shell
cd examples
python3 cnn1d_iscx_session_all_train.py
```

After execution, the default checkpoint file will be generated in ./models_ckpt/cnn1d, and the inference
checkpoint file will be generated in ./models_eval_ckpt/cnn1d. You can also modify the checkpoint_path
and checkpoint_eval_path of the cnn1d_iscx_session_all_train.py file to change the generated file path.

## Convert the teacher model to logits output
Enter the tools directory and execute:
```shell
cd tools
python3 convert_softmax_model_to_logits.py
```

After execution, the default checkpoint file of logits model will be generated in examples/models_eval_ckpt/cnn1d/
checkpoint-60-logits.h5

## Create the distilling process and distill the cnn1d_tiny model
Create the configuration file in the src/model_optimizer/pruner/scheduler/distill,like "cnn1d_tiny_0.3.yaml" where the 
distillation parameters is configured.

Enter the examples directory, create cnn1d_tiny_iscx_session_all_distill.py for cnn1d_tiny model. In the distilling 
process, the teacher is cnn1d, the student is cnn1d_tiny. 

> Note
>  
> > the "model_name" and "dataset" in the request must be the same as you defined before

```shell
python3 cnn1d_tiny_iscx_session_all_distill.py
```

After execution, the default checkpoint file will be generated in ./models_ckpt/cnn1d_tiny_distill, and the inference 
checkpoint file will be generated in ./models_eval_ckpt/cnn1d_tiny_distill. You can also  modify the checkpoint_path and
checkpoint_eval_path of the cnn1d_tiny_iscx_session_all_distill.py file to change the 
generated file path.

> Note
>  
> > i. The model in the checkpoint_path is not the pure cnn1d_tiny model. It's the hybird of cnn1d_tiny(student) and  
> > cnn1d(teacher)
> >
> > ii. The model in the checkpoint_eval_path is the distilled model, i.e. pure cnn1d_tiny model
