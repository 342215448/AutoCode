import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoModelForImageClassification
import warnings
warnings.filterwarnings("ignore")

# 进行图像预处理
def load_image(image_path, mode_size):

    picture_dataSize = int(mode_size)
    transform = transforms.Compose([
        transforms.Resize((picture_dataSize, picture_dataSize)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

def inference_image(model_path, image_path, folder_path, pretrain_model):

    # 加载图像
    image = load_image(image_path, model_path.split('/')[-1].split('_')[-3].split('-')[-1])

    # 初始化模型
    model = AutoModelForImageClassification.from_pretrained(pretrain_model)

    # 加载模型参数
    # model.load_state_dict(torch.load(model_path))
    # 解决多卡训练的模型单卡推理时外层嵌套model.的问题
    weights = torch.load(model_path)
    
    weights_dict = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v
    
    model.load_state_dict(weights_dict)

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


if __name__ == "__main__":
    # 提供模型字典的参数
    model_path = "checkpoints/beit-base-224_JYX_best-model.pth"

    # 输入图像的路径
    image_path = "dataset/JYX/test/JYX2/JYX2 (401).JPG"

    # 设置文件夹路径
    folder_path = image_path.split('/')[0]+image_path.split('/')[1]+image_path.split('/')[2]
    # folder_path = "dataset/JYX/test/"

    # 设置模型路径
    pretrain_model = "models/base-model/"+model_path.split('/')[-1].split('_')[-3]

    # 执行推理
    inference_image(model_path, image_path, folder_path, pretrain_model)
