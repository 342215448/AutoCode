import os
import matplotlib.pyplot as plt

# 定义文件路径
folder_path = "loss_data"


def plotLoss(folder_path):
    # 遍历文件夹中的所有.txt文件
    for file_name in os.listdir(folder_path):
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

            # 绘制曲线
            plt.plot(loss_values, label=label)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 添加标题和坐标轴标签
    plt.title("Loss function changes")
    plt.xlabel("training steps")
    plt.ylabel("Loss function value")

    # 添加图例
    plt.legend()

    # 显示图表
    plt.show()


def plotAcc(folder_path):
    # 遍历文件夹中的所有.txt文件
    for file_name in os.listdir(folder_path):
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
    plt.title("Loss function changes")
    plt.xlabel("training steps")
    plt.ylabel("Loss function value")

    # 添加图例
    plt.legend()

    # 显示图表
    plt.show()


if __name__ == "__main__":
    #  绘制loss曲线
    plotLoss(folder_path)
    #  绘制acc曲线
