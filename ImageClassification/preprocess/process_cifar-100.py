import os
import pickle
import numpy as np
from PIL import Image
from tqdm.contrib import tzip

# 设置数据集路径和输出路径
dataset_path = '/mnt/muyongyu/fpn/wrq/example_project/dataset/cifar-100-python'
output_path = '/mnt/muyongyu/fpn/wrq/example_project/dataset/cifar-100'

# 创建输出文件夹
os.makedirs(output_path, exist_ok=True)

# 加载数据集


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

# 转换图像并保存到对应类别文件夹


def convert_images(data, output_path):
    for i, (image, label) in enumerate(tzip(data[b'data'], data[b'coarse_labels'])):
        # 重新构造图像数组的形状
        image = np.transpose(np.reshape(image, (3, 32, 32)), (1, 2, 0))
        # 创建图像对象
        img = Image.fromarray(image)
        # 创建类别文件夹
        label_path = os.path.join(output_path, str(label))
        os.makedirs(label_path, exist_ok=True)
        # 保存图像
        img.save(os.path.join(label_path, f'image_{i}.png'))


# 处理训练集数据
train_path = os.path.join(dataset_path, 'train')
train_data = unpickle(train_path)
train_output_path = os.path.join(output_path, 'train')
os.makedirs(train_output_path, exist_ok=True)
convert_images(train_data, train_output_path)

# 处理测试集数据
test_path = os.path.join(dataset_path, 'test')
test_data = unpickle(test_path)
test_output_path = os.path.join(output_path, 'test')
os.makedirs(test_output_path, exist_ok=True)
convert_images(test_data, test_output_path)

print("转换完成！")
