import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoModelForImageClassification
import warnings
warnings.filterwarnings("ignore")

# 进行图像预处理
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image


if __name__ == "__main__":
    # 提供模型字典的参数
    model_path = "/workspace/ImageClassification/checkpoints/beit-base-224_cifar-10-batchesbest_model.pth"

    # 输入图像的路径
    image_path = "/workspace/ImageClassification/dataset/cifar-10-batches/test/ship/image_72.png"

    # 设置文件夹路径
    folder_path = "/workspace/ImageClassification/dataset/cifar-10-batches/test"

    # 设置模型路径
    pretrain_model = "/workspace/ImageClassification/models/base-model/beit-base-224"

    # 加载图像
    image = load_image(image_path)

    # 初始化模型
    model = AutoModelForImageClassification.from_pretrained(pretrain_model)

    # 加载模型参数
    model.load_state_dict(torch.load(model_path))

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)

    # 进行预测
    outputs = model(image)
    probabilities = torch.softmax(outputs.logits, dim=1)
    labels = sorted(os.listdir(folder_path))
    results = [(labels[i], probabilities[0, i].item()) for i in range(len(labels))]

    # 打印各概率的值
    for label, probability in results:
        print(f"{label}: {probability:.4f}")

    # 输出可能性最高的结果
    results.sort(key=lambda x: x[1], reverse=True)
    top_label, top_probability = results[0]
    print(f"Top Prediction: {top_label} ({top_probability:.4f})")