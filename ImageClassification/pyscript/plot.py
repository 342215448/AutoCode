import os
import random
import matplotlib.pyplot as plt
from itertools import cycle


# 创建文件夹的函数
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 创建成功")
    else:
        print(f"文件夹 '{folder_path}' 已经存在")


def plotLoss(folder_path, save_path, n):
    # 遍历文件夹中的所有.txt文件
    for file_name, style in sorted(zip(os.listdir(folder_path), cycle(['-', '--', ':']))):
        if file_name.endswith(".txt"):
            # 构建文件的完整路径
            file_path = os.path.join(folder_path, file_name)

            # 读取.txt文件
            with open(file_path, "r") as file:
                lines = file.readlines()

            # 提取损失函数数据
            loss_values = [float(lines[i].strip()) for i in range(0, len(lines), n)]

            # 使用文件名作为曲线标签
            label = os.path.splitext(file_name)[0]

            # 绘制曲线
            plt.plot(loss_values, label=label, linestyle=style)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 添加标题和坐标轴标签
    plt.title("Loss Changes")
    plt.xlabel("steps")
    plt.ylabel("loss value")

    # 添加图例
    plt.legend()

    # 保存图片
    file_path = os.path.join(save_path, 'loss_plot.png')
    plt.savefig(file_path)

    # 显示图表 在本机上可选
    # plt.show()


def plotAcc(folder_path, save_path):
    if folder_path.split('/')[-1] == 'train_acc':
        title_name = 'Train Accuracy Changes'
        xlabel_name = 'train'
    else:
        title_name = 'Validation Accuracy Changes'
        xlabel_name = 'validation'
    # 遍历文件夹中的所有.txt文件
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".txt"):
            # 构建文件的完整路径
            file_path = os.path.join(folder_path, file_name)

            # 读取.txt文件
            with open(file_path, "r") as file:
                lines = file.readlines()

            # 提取损失函数数据
            loss_values = [float(line.strip()) for line in lines]

            # 使用文件名作为曲线标签
            label = os.path.splitext(file_name)[0]

            # 随机选择符号标记
            markers = ['^', 'o']  # 可以根据需要添加更多标记
            marker = random.choice(markers)

            # 绘制曲线
            plt.plot(loss_values, label=label, marker=marker)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 添加标题和坐标轴标签
    plt.title(title_name)
    plt.xlabel("steps")
    plt.ylabel(xlabel_name + " accuracy value")

    # 添加图例
    plt.legend()

    # 保存图片
    file_path = os.path.join(save_path, xlabel_name + '_acc_plot.png')
    plt.savefig(file_path)

    # 显示图表 在本机上可选
    plt.show()


if __name__ == "__main__":
    # 定义源文件路径
    src_path = "plot/base-model"
    # 设置结果保存路径
    loss_folder_path = 'result/loss_output'
    acc_folder_path = 'result/acc_output'
    # 设置loss图像的平滑度 值越平滑
    smooth_level = 2

    for file_name in os.listdir(src_path):
        folder_path = src_path + '/' + file_name
        # print(folder_path)
        create_folder_if_not_exists(loss_folder_path)
        create_folder_if_not_exists(acc_folder_path)
        if folder_path.split('/')[-1] == 'loss':
            # 绘制loss曲线
            plotLoss(folder_path, loss_folder_path, smooth_level)
        else:
            # 绘制acc曲线
            plotAcc(folder_path, acc_folder_path)
