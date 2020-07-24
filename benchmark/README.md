# About the benchmark

The benchmark is used to test  the adlik serving performance of different models which are optimized by model_optimizer.

## Test the runtime performance

Take ResNet-50 as an example to illustrate how to test the model performance

1. clone model_optimizer  and change the working directory into the source directory

   ```sh
   git clone https://github.com/Adlik/model_optimizer.git
   cd model_optimizer
   ```

2. make a model directory and put resnet50.h5 in it

   ```sh
   mkdir -p benchmark/tests/test_model/resnet50_keras/model  
   ```

3. change the working directory into benchmark tools directory, execute the shell script automatic_test.sh, which can auto test the performance of ResNet50

   ```sh
   cd benchmark/tools/resnet50
   ./automatic_test.sh
   ```

4. you can also test the performance of optimized model of ResnNet-50. Put the  resnet50_pruned.h5 under model directory  and execute the shell script automatic_test_pruned.sh
