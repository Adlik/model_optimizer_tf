#!/usr/bin/env bash
cd ../../../../
git clone https://github.com/Adlik/Adlik.git
base_path=$(cd `dirname $0`; pwd)
echo "base_path: "$base_path
optimizer_benchmark_path=$base_path/model_optimizer/benchmark
adlik_benchmark_path=$base_path/Adlik/benchmark
adlik_path=$base_path/Adlik 
cp -r $optimizer_benchmark_path/tests $adlik_benchmark_path

cd $adlik_path
echo "*****************start to make benchmark image and test****************************"
python3 benchmark/src/automatic_test.py -d benchmark/tests/docker_test/tensorflow_gpu.Dockerfile -s tensorflow_gpu -b . -a . -m resnet50_pruned  -c benchmark/tests/client_script/resnet50_pruned_client_script.sh -ss benchmark/tests/serving_script/resnet50_pruned_serving_script.sh -l /home/pjy/adlik/Adlik/benchmark/log -tm benchmark/tests/test_model/resnet50_keras -cis imagenet_client.py -i imagenet.JPEG -cs benchmark/tests/compile_script/compile_script.sh  -sj resnet50_pruned_serving_model.json -gl 0
