#!/bin/bash

# 选择使用多张gpu 在使用前先用gpustat看看有几张卡可以用，然后对应使用多gpu
export CUDA_VISIBLE_DEVICES=0
# 模型文件路径
models_path="models"
# 设置数据集路径
data_path="dataset"
# 训练的轮数
epochs=1
# train or test
mode="train"
# batchsize 32 64 128 max ...
bs=32

echo "Now is on ${mode} mode..."

# 自动匹配models_path路径下所有的模型
for dir in "$models_path"/*; do
    if [ -d "$dir" ]; then
        for src in "$data_path"/*; do
            if [ -d "$src" ]; then
                # 只处理目录
                dirname=$(basename "$dir")
                dataname=$(basename "$src")
                echo "#################   exciting ${dirname} with ${dataname} now   #################"
                # python pyscript/eachdataset_eachmodel.py --models_path "$models_path/$dirname" --mode "$mode" --epochs "$epochs" --dataset_path "$data_path/$dataname"
                # echo "python eachdataset_eachmodel.py --models_path "$models_path$dirname" --mode "$mode" --epochs "$epochs" --dataset_path "$data_path/$dataname""
                python pyscript/train.py --models_path "$models_path/$dirname" --data_path "$data_path/$dataname" --mode "$mode" --epochs "$epochs" --batchsize "$bs"
            fi
        done
    fi
done
echo "done..."