#!/bin/bash

# # 模型文件路径
# models_path="models/base-model"
# data_path="dataset/JYX"


# # 训练的轮数
# epochs=1
# # train or test
# mode="test"
# # batchsize 32 64 128 max....
# bs=32

# echo "Now is on ${mode} mode..."
# # 自动匹配models_path路径下所有的模型
# for dir in "$models_path"/*; do
#     if [ -d "$dir" ]; then
#         # 只处理目录
#         dirname=$(basename "$dir")
#         echo "#################   exciting ${dirname} now   #################"
#         python pyscript/train.py --models_path "$models_path/$dirname" --data_path "$data_path" --mode "$mode" --epochs "$epochs" --batchsize "$bs"
#     fi
# done
# 提供模型字典的参数
    # model_path = "/workspace/ImageClassification/checkpoints/beit-base-224_cifar-10-batchesbest_model.pth"

    # # 输入图像的路径
    # image_path = "/workspace/ImageClassification/dataset/cifar-10-batches/test/ship/image_72.png"

    # # 设置文件夹路径
    # folder_path = "/workspace/ImageClassification/dataset/cifar-10-batches/test"

    # # 设置模型路径
    # pretrain_model = "/workspace/ImageClassification/models/base-model/beit-base-224"

    # # 执行推理
    # inference_image(model_path, image_path, folder_path, pretrain_model)
model_path=
image_path=
folder_path=
pretrain_model=

echo "Done..."