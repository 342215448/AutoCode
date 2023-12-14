import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from transformers import AutoModelForImageClassification, get_scheduler
from tqdm import tqdm
import argparse
import time
from torch.cuda.amp import autocast, GradScaler

import warnings
warnings.filterwarnings("ignore")


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

# 训练函数
def train_epoch(model, loader, criterion, optimizer, output_file, is_warmup, gc, scaler, is_autocast, total):
    model.train()
    running_loss = 0.0
    correct_predictions = 0

    if is_warmup:
        # 设置学习率策略
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0.1,
            num_training_steps=total,
        )

    count_ = 0
    for images, labels in tqdm(loader, desc="trainging..."):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if is_autocast:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs.logits, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
        output_file.write('{:.2f}'.format(loss.item()))
        output_file.write('\n')
        if is_autocast:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        count_ += 1
        if count_ == gc:
            if is_autocast:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                optimizer.zero_grad()
            count_ = 0
        if is_warmup:
            lr_scheduler.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.logits, 1)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    accuracy = correct_predictions / len(loader.dataset)

    return epoch_loss, accuracy

# 验证函数
def eval_epoch(model, loader, criterion, is_autocast):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="evaluating..."):
            images = images.to(device)
            labels = labels.to(device)

            if is_autocast:
                with autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            loss = criterion(outputs.logits, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    accuracy = correct_predictions / len(loader.dataset)

    return epoch_loss, accuracy

# 测试函数
def test_epoch(model, loader, criterion, is_autocast):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="testing..."):
            images = images.to(device)
            labels = labels.to(device)

            if is_autocast:
                with autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            loss = criterion(outputs.logits, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    accuracy = correct_predictions / len(loader.dataset)

    return epoch_loss, accuracy

# 训练函数
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_name, data_name, mode_name, is_warmup, gc, scaler, is_autocast, total):
    best_val_accuracy = 0.0
    loss_path = 'plot/'+mode_name+'/loss/'
    train_acc_path = 'plot/'+mode_name+'/train_acc/'
    val_acc_path = 'plot/'+mode_name+'/val_acc/'
    create_folder_if_not_exists(loss_path)
    create_folder_if_not_exists(train_acc_path)
    create_folder_if_not_exists(val_acc_path)
    output_file = open(loss_path+model_name+'-'+data_name+'-loss.txt', 'w')
    output_acc = open(train_acc_path+model_name + '-'+data_name+'-trainacc.txt', 'w')
    output_valacc = open(val_acc_path+model_name + '-'+data_name+'-valacc.txt', 'w')
    print("Using "+model_name+" with "+data_name)

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        train_loss, train_accuracy = train_epoch(
            model, train_loader, criterion, optimizer, output_file, is_warmup, gc, scaler, is_autocast, total)
        #  写入train_acc
        output_acc.write('{:.2f}'.format(train_accuracy))
        output_acc.write('\n')
        print(
            f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

        val_loss, val_accuracy = eval_epoch(model, val_loader, criterion, is_autocast)
        #  写入val_acc
        output_valacc.write('{:.2f}'.format(val_accuracy))
        output_valacc.write('\n')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # 创建checkpoints路径
        create_folder_if_not_exists('checkpoints/')
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'checkpoints/' +
                       model_name + '_' +data_name + '_best-model.pth')
            print('Saved the best model.')

        print('--------------------------------------')

    print('Training finished.')
    output_file.close()
    output_acc.close()
    output_valacc.close()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f'运行时间为:{execution_time/60}min')


# 测试函数
def batch_test(val_loader, criterion, model_name, data_name, mode_name, is_autocast):
    test_acc_path = 'plot/'+mode_name+'/test_acc/'
    create_folder_if_not_exists(test_acc_path)
    # 加载模型
    model.load_state_dict(torch.load('checkpoints/'+model_name+'_'+data_name + '_best-model.pth'))
    print('Best model has been loaded...')
    # 打开文本文件以追加模式写入
    output_file = open(test_acc_path+'test_output.csv', 'a', newline='')  
    csv_writer = csv.writer(output_file)
    #output_file.write('--------------------------------------\n')
    #output_file.write('checkpoints/' + model_name + '_' +data_name +'best_model.pth'+' has been loaded...\n')
    val_loss, val_accuracy = test_epoch(model, val_loader, criterion, is_autocast)

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
    parser.add_argument("--models_path", type=str, default='models/base-model/swin-base-384',
                        help="Path of models, defaults resnet-50.")
    parser.add_argument("--mode", type=str, default='train',
                        help="train or test?")
    parser.add_argument("--epochs", type=int, default=20,
                        help="The number of epochs to train.")
    parser.add_argument("--batchsize", type=int, default=16,
                        help="The batchsize. If you want larger batchsize and you don't have more mem, try --gradientAccu larger...")
    parser.add_argument("--data_path", type=str, default='data/score',
                        help="Path of dataset.")
    parser.add_argument("--fastTrain", type=bool, default=True,
                        help="When your mem is big enough...")
    parser.add_argument("--gradientAccu", type=int, default=4,
                        help="Use gradient accumulation to make total batchsize bigger...")
    parser.add_argument("--autocast", type=bool, default=True,
                        help="Use mixed precision computation to accelerate trian speed...")
    parser.add_argument("--dlr", type=bool, default=False,
                        help="Use dynamic learning rate...")
    args = parser.parse_args()
    model_path = args.models_path
    mode = args.mode
    num_epochs = args.epochs

    # 路径设置(数据路径)
    data_path = args.data_path
    train_path = data_path+'/train'
    test_path = data_path+'/test'

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
    # 判断一下是否将全部数据放置于内存提高训练速度
    if args.fastTrain:
        train_list = []
        val_list = []
        for line in tqdm(train_dataset, desc='loading train dataset...'):
            train_list.append(line)
        for line in tqdm(val_dataset, desc='loading validation dataset...'):
            val_list.append(line)
        # 创建数据加载器
        batch_size = args.batchsize
        # 根据线程数设置多线程处理数据 pin_memory在使用gpu的情况下要开启作为加速
        num_workers = torch.get_num_threads()
        
        train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        # 创建数据加载器
        batch_size = args.batchsize
        # 根据线程数设置多线程处理数据 pin_memory在使用gpu的情况下要开启作为加速
        num_workers = torch.get_num_threads()
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 初始化模型
    model = AutoModelForImageClassification.from_pretrained(model_path)
    model.to(device)
    # 使用多gpu
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))
        model = nn.DataParallel(model)

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-5/args.gradientAccu)
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    scaler = GradScaler()
    if mode == 'train':
        train(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_path.split('/')[-1], data_path.split('/')[-1], model_path.split('/')[-2], args.dlr, args.gradientAccu, scaler, args.autocast, len(train_loader)*num_epochs)
    elif mode == 'test':
        batch_test(test_loader, criterion, model_path.split('/')[-1], data_path.split('/')[-1], model_path.split('/')[-2], args.autocast)
    else:
        print("sorry, you have to choose one mode of train or test...")

# python pyscript/classifier.py --models_path /workspace/ImageClassification/models/beit-base-patch16-224 --data_path /workspace/ImageClassification/dataset/cifar-10-batches --mode train --epochs 10 --batchsize 32
# python pyscript/classifier.py --models_path /workspace/ImageClassification/models/beit-base-patch16-224 --data_path /workspace/ImageClassification/dataset/cifar-10-batches --mode test --batchsize 32
