#!/bin/bash

# 选择使用多张gpu 在使用前先用gpustat看看有几张卡可以用，然后对应使用多gpu
export CUDA_VISIBLE_DEVICES=0
# 模型文件路径
models_path="models/base-model"
data_path="dataset/JYX"


# 训练的轮数
epochs=1
# train or test
mode="test"
# batchsize 32 64 128 max....
bs=32

echo "Now is on ${mode} mode..."
# 自动匹配models_path路径下所有的模型
for dir in "$models_path"/*; do
    if [ -d "$dir" ]; then
        # 只处理目录
        dirname=$(basename "$dir")
        echo "#################   exciting ${dirname} now   #################"
        python pyscript/classifier.py --models_path "$models_path/$dirname" --data_path "$data_path" --mode "$mode" --epochs "$epochs" --batchsize "$bs"
    fi
done
echo "Done..."