# About the benchmark

The benchmark is used to test the adlik serving performance of different models which are optimized by model_optimizer tf.

## Test the runtime performance

Take ResNet-50 as an example to illustrate how to test the model performance

1. clone model_optimizer tf and change the working directory into the source directory

   ```sh
   git clone https://github.com/Adlik/model_optimizer_tf.git
   cd model_optimizer_tf
   ```

2. make a model directory and put resnet50.h5 in it

   ```sh
   mkdir -p benchmark/tests/test_model/resnet50_keras/model  
   ```

3. change the working directory into benchmark tools directory, execute the shell script automatic_test.sh, which can
auto test the performance of ResNet50

   ```sh
   cd benchmark/tools/resnet50
   ./automatic_test.sh
   ```

4. you can also test the performance of optimized model of ResnNet-50. Put the  resnet50_pruned.h5 under model directory
 and execute the shell script automatic_test_pruned.sh

## The test result of the model in keras-tfGPU2.1

|                  | speed of client (pictures/sec) | speed of serving engine (pictures/sec) | tail latency of one picture (sec) |
| ---------------- | :----------------------------: | :------------------------------------: | :-------------------------------: |
| ResNet-50        |            184.872             |                480.882                 |              0.00333              |
| ResNet-50-L1-0.3 |            191.531             |                518.280                 |              0.00329              |

> Note
>
> > i. ResNet-50 is the baseline of training
> >
> > ii. ResNet-50-L1-0.3 denotes ResNet-50 model use L1 norm filter pruning, pruning rate is 30%
