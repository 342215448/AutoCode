import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from transformers import AutoModelForImageClassification
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings("ignore")

learning_rate = 1e-6

# 设置设备为GPU（如果可用），否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# 创建文件夹的函数
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 创建成功")
    # else:
    #     print(f"文件夹 '{folder_path}' 已经存在")


# 多线程转换函数
def multiThread_convert(train_loader):
    # 提前把图片数据放入gpu
    loader_list = []
    count_num = 0
    for i in tqdm(enumerate(train_loader), desc='processing to gpu...'):
        loader_list.append([i[1][0], i[1][1]])
        loader_list[count_num][0] = loader_list[count_num][0].to(device)
        loader_list[count_num][1] = loader_list[count_num][1].to(device)
        count_num += 1
    return loader_list


# 训练函数
def train_epoch(model, loader_list, criterion, optimizer, output_file, train_loader):
    model.train()
    running_loss = 0.0
    correct_predictions = 0

    # for images, labels in tqdm(loader, desc="trainging..."):
    for i in tqdm(loader_list, desc='training...'):
        images = i[0]
        labels = i[1]

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        output_file.write('{:.2f}'.format(loss.item()))
        output_file.write('\n')
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.logits, 1)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    accuracy = correct_predictions / len(train_loader.dataset)

    return epoch_loss, accuracy


# 验证函数
def eval_epoch(model, loader, criterion, loader_list):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0

    with torch.no_grad():
        # for images, labels in tqdm(loader, desc="evaluating..."):
        for i in tqdm(loader_list, desc='training...'):
            images = i[0]
            labels = i[1]

            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    accuracy = correct_predictions / len(loader.dataset)

    return epoch_loss, accuracy


# 测试函数
def test_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="testing..."):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    accuracy = correct_predictions / len(loader.dataset)

    return epoch_loss, accuracy


# 训练函数
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_name, data_name, mode_name):
    best_val_accuracy = 0.0
    loss_path = 'plot/' + mode_name + '/loss/'
    train_acc_path = 'plot/' + mode_name + '/train_acc/'
    val_acc_path = 'plot/' + mode_name + '/val_acc/'
    create_folder_if_not_exists(loss_path)
    create_folder_if_not_exists(train_acc_path)
    create_folder_if_not_exists(val_acc_path)
    output_file = open(loss_path + model_name + '-' + data_name + '-loss.txt', 'w')
    output_acc = open(train_acc_path + model_name + '-' + data_name + '-trainacc.txt', 'w')
    output_valacc = open(val_acc_path + model_name + '-' + data_name + '-valacc.txt', 'w')
    print("Using " + model_name + " with " + data_name)
    num_pres_epoch = 1

    # 提前把图片数据放入gpu
    trainloader_list = []
    count_num = 0
    for i in tqdm(enumerate(train_loader), desc='processing train to gpu...', total=len(train_loader)):
        trainloader_list.append([i[1][0], i[1][1]])
        trainloader_list[count_num][0] = trainloader_list[count_num][0].to(device)
        trainloader_list[count_num][1] = trainloader_list[count_num][1].to(device)
        count_num += 1

    valloader_list = []
    count_num = 0
    for i in tqdm(enumerate(val_loader), desc='processing validation to gpu...', total=len(val_loader)):
        valloader_list.append([i[1][0], i[1][1]])
        valloader_list[count_num][0] = valloader_list[count_num][0].to(device)
        valloader_list[count_num][1] = valloader_list[count_num][1].to(device)
        count_num += 1
    for epoch in range(num_epochs):
        # if num_pres_epoch == 10:
        #     optimizer = optim.Adam(model.parameters(), lr=learning_rate*0.5)
        #     print(f'LearningRate changed to {learning_rate*0.5}...')
        # if num_pres_epoch == 20:
        #     optimizer = optim.Adam(model.parameters(), lr=learning_rate*0.1)
        #     print(f'LearningRate changed to {learning_rate*0.1}...')
        # if num_pres_epoch == 30:
        #     optimizer = optim.Adam(model.parameters(), lr=learning_rate*0.01)
        #     print(f'LearningRate changed to {learning_rate*0.01}...')
        print(f'Epoch {epoch + 1}/{num_epochs}')
        train_loss, train_accuracy = train_epoch(
            model, trainloader_list, criterion, optimizer, output_file, train_loader)
        #  写入train_acc
        output_acc.write('{:.2f}'.format(train_accuracy))
        output_acc.write('\n')
        print(
            f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

        val_loss, val_accuracy = eval_epoch(model, val_loader, criterion, valloader_list)
        #  写入val_acc
        output_valacc.write('{:.2f}'.format(val_accuracy))
        output_valacc.write('\n')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # 创建checkpoints路径
        create_folder_if_not_exists('checkpoints/')
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'checkpoints/' +
                       model_name + '_' + data_name + '_best-model.pth')
            print('Saved the best model.')

        print('--------------------------------------')
        num_pres_epoch += 1

    print('Training finished.')
    output_file.close()
    output_acc.close()
    output_valacc.close()


# 测试函数
def batch_test(val_loader, criterion, model_name, data_name, mode_name):
    test_acc_path = 'plot/' + mode_name + '/test_acc/'
    create_folder_if_not_exists(test_acc_path)
    # 加载模型
    model.load_state_dict(torch.load('checkpoints/' + model_name + '_' + data_name + '_best-model.pth'))
    print('Best model has been loaded...')
    # 打开文本文件以追加模式写入
    output_file = open(test_acc_path + 'test_output.csv', 'a', newline='')
    csv_writer = csv.writer(output_file)
    # output_file.write('--------------------------------------\n')
    # output_file.write('checkpoints/' + model_name + '_' +data_name +'best_model.pth'+' has been loaded...\n')
    val_loss, val_accuracy = test_epoch(model, val_loader, criterion)

    # output_file.write(f'model_name {val_accuracy:.2f}\n')
    csv_writer.writerow([model_name, f'{val_accuracy:.2f}'])
    # output_file.write('--------------------------------------\n\n\n')
    output_file.close()  # 关闭文件

    print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    print('--------------------------------------')
    print('Batch testing finished.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate articles using a pretrained model.")
    parser.add_argument("--models_path", type=str,
                        help="Path of models, defaults resnet-50.", default='../models/base-model/swin-base-384')
    parser.add_argument("--mode", type=str,
                        help="train or test?", default='train')
    parser.add_argument("--epochs", type=int,
                        help="The number of epochs to train.", default=20)
    parser.add_argument("--batchsize", type=int,
                        help="The batchsize.", default=16)
    parser.add_argument("--data_path", type=str,
                        help="Path of dataset.", default='../data/all_score')
    parser.add_argument("--lr", type=float,
                        help="Learning rate.", default=0.00005)

    args = parser.parse_args()
    model_path = args.models_path
    mode = args.mode
    num_epochs = args.epochs
    learning_rate = args.lr

    # 路径设置(数据路径)
    data_path = args.data_path
    train_path = data_path + '/train'
    test_path = data_path + '/test'

    # 数据预处理
    picture_dataSize = int(model_path.split('/')[-1].split('-')[-1])
    transform = transforms.Compose([
        transforms.Resize((picture_dataSize, picture_dataSize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载训练集和验证集
    train_dataset = ImageFolder(train_path, transform=transform)
    test_dataset = ImageFolder(test_path, transform=transform)

    # 划分训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size])

    # 创建数据加载器
    batch_size = args.batchsize
    # 根据线程数设置多线程处理数据 pin_memory在使用gpu的情况下要开启作为加速
    num_workers = torch.get_num_threads()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    # 初始化模型
    model = AutoModelForImageClassification.from_pretrained(model_path)
    model.to(device)
    # 使用多gpu
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))
        model = nn.DataParallel(model)

    # 定义优化器 Adamw
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    if mode == 'train':
        train(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_path.split('/')[-1],
              data_path.split('/')[-1], model_path.split('/')[-2])
    elif mode == 'test':
        batch_test(test_loader, criterion, model_path.split('/')[-1], data_path.split('/')[-1],
                   model_path.split('/')[-2])
    else:
        print("sorry, you have to choose one mode of train or test...")

# python pyscript/train.py --models_path models/base-model/beit-base-224 --data_path dataset/JYB --mode train --epochs 50 --batchsize 16 --lr 0.00001
