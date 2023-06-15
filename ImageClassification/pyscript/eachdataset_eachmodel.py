import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from transformers import AutoModelForImageClassification
from tqdm import tqdm
import argparse

# 路径设置(包括训练数据路径、测试数据路径、预训练模型路径)
# dataset_path = '/mnt/muyongyu/fpn/wrq/example_project/dataset/cifar-10-batches'
# pretrainedModel_path = '/mnt/muyongyu/fpn/wrq/example_project/models/resnet-50'
# train_path = dataset_path+'/train'
# test_path = dataset_path+'/test'

# 设置设备为GPU（如果可用），否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 训练函数


def train_epoch(model, loader, criterion, optimizer, output_file):
    model.train()
    running_loss = 0.0
    correct_predictions = 0

    for images, labels in tqdm(loader, desc="trainging..."):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        output_file.write('{:.2f}'.format(loss.item()))
        output_file.write('\n')
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.logits, 1)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    accuracy = correct_predictions / len(loader.dataset)

    return epoch_loss, accuracy

# 验证函数


def eval_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="evaluating..."):
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


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_name, data_name):
    best_val_accuracy = 0.0
    output_file = open('plot/loss/'+model_name+'-'+data_name+'loss.txt', 'w')
    output_acc = open('plot/train_acc/'+model_name +
                      '-'+data_name+'trainacc.txt', 'w')
    output_valacc = open('plot/val_acc/'+model_name +
                         '-'+data_name+'valacc.txt', 'w')
    print("Using "+model_name+" with "+data_name)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        train_loss, train_accuracy = train_epoch(
            model, train_loader, criterion, optimizer, output_file)
        #  写入train_acc
        output_acc.write('{:.2f}'.format(train_accuracy))
        output_acc.write('\n')
        print(
            f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

        val_loss, val_accuracy = eval_epoch(model, val_loader, criterion)
        #  写入val_acc
        output_valacc.write('{:.2f}'.format(val_accuracy))
        output_valacc.write('\n')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'checkpoints/' +
                       model_name + data_name + 'best_model.pth')
            print('Saved the best model.')

        print('--------------------------------------')

    print('Training finished.')
    output_file.close()
    output_acc.close()
    output_valacc.close()


def batch_test(val_loader, criterion, model_name, data_name):
    model.load_state_dict(torch.load(
        'checkpoints/'+model_name + data_name + 'best_model.pth'))
    print('Best model has been loaded...')
    output_file = open(
        'result/multi_datasets/batchtest_output.txt', 'a')  # 打开文本文件以追加模式写入
    output_file.write('--------------------------------------\n')
    output_file.write('checkpoints/'+model_name+data_name +
                      'best_model.pth'+' has been loaded...\n')
    val_loss, val_accuracy = test_epoch(model, val_loader, criterion)

    output_file.write(
        f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}\n')
    output_file.write('--------------------------------------\n\n\n')
    output_file.close()  # 关闭文件

    print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    print('--------------------------------------')
    print('Batch testing finished.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate articles using a pretrained model.")
    parser.add_argument("--models_path", type=str,
                        help="Path of models, defaults resnet-50.", required=True)
    parser.add_argument("--mode", type=str,
                        help="train or test?")
    parser.add_argument("--epochs", type=int,
                        help="The number of epochs to train.")
    parser.add_argument("--dataset_path", type=str,
                        help="Path of datasets, defaults cifar-10-batches.", required=True)
    args = parser.parse_args()
    model_path = args.models_path
    mode = args.mode
    num_epochs = args.epochs
    train_path = args.dataset_path+'/train'
    test_path = args.dataset_path+'/test'

    # 加载训练集和验证集
    train_dataset = ImageFolder(train_path, transform=transform)
    test_dataset = ImageFolder(test_path, transform=transform)

    # 划分训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size])

    # 创建数据加载器
    batch_size = 64
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化ViT模型
    # model = ViTForImageClassification.from_pretrained(pretrainedModel_path)
    model = AutoModelForImageClassification.from_pretrained(model_path)
    model.to(device)
    # 使用多dp
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))
        model = nn.DataParallel(model)

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    pretrainedModel_path = model_path
    if mode == 'train':
        train(model, train_loader, val_loader,
              criterion, optimizer, num_epochs, model_path.split('/')[-1], args.dataset_path.split('/')[-1])
    elif mode == 'test':
        batch_test(test_loader, criterion, model_path.split(
            '/')[-1], args.dataset_path.split('/')[-1])
    else:
        print("sorry, you have to choose one mode of train or test...")

# python eachdataset_eachmodel.py --models_path /mnt/muyongyu/fpn/wrq/example_project/models/resnet-50 --mode train --epochs 10 --dataset_path /mnt/muyongyu/fpn/wrq/example_project/dataset/cifar-10-batches
# python eachdataset_eachmodel.py --models_path /mnt/muyongyu/fpn/wrq/example_project/models/resnet-50 --mode test --dataset_path /mnt/muyongyu/fpn/wrq/example_project/dataset/cifar-10-batches/
