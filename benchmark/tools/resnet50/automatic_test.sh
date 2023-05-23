#!/usr/bin/env bash
current_path=$(cd `dirname $0`; pwd)
echo "current_path: "$current_path
base_path=$current_path/../../../..
echo "base_path: "$base_path
cd $base_path
git clone https://github.com/Adlik/Adlik.git
optimizer_benchmark_path=$base_path/model_optimizer_tf/benchmark
adlik_benchmark_path=$base_path/Adlik/benchmark
adlik_path=$base_path/Adlik 
cp -r $optimizer_benchmark_path/tests $adlik_benchmark_path

cd $adlik_path
echo "*****************start to make benchmark image and test****************************"
python3 benchmark/src/automatic_test.py -d benchmark/tests/docker_test/tensorflow_gpu.Dockerfile -s tensorflow_gpu -b . -a . -m resnet50  -c benchmark/tests/client_script/resnet50_client_script.sh -ss benchmark/tests/serving_script/resnet50_serving_script.sh -l /home/pjy/adlik/Adlik/benchmark/log -tm benchmark/tests/test_model/resnet50_keras -cis imagenet_client.py -i imagenet.JPEG -cs benchmark/tests/compile_script/compile_script.sh  -sj resnet50_serving_model.json -gl 0
