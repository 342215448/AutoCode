#!/bin/bash

# 选择使用多张gpu 在使用前先用gpustat看看有几张卡可以用，然后对应使用多gpu
export CUDA_VISIBLE_DEVICES=0
# 模型文件路径
models_path="models/base-model/vit-base-224"
data_path="data/all_score"
chl_data="data/step1/secai_224.chl"


# 训练的轮数
epochs=15
# train or test
mode="train"
# batchsize 32 64 128 max....
bs=192

python pyscript/train.py --models_path "$models_path" --data_path "$data_path" --mode "$mode" --epochs "$epochs" --batchsize "$bs" --chldata  "$chl_data"
# accelerate launch --config_file acc_config.yaml pyscript/train.py --models_path "$models_path" --data_path "$data_path" --mode "$mode" --epochs "$epochs" --batchsize "$bs" --chldata  "$chl_data"
echo "Done..."


# accelerate launch --config_file acc_config.yaml train_diamond_miniimagenet_fpn.py
