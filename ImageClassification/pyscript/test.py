import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# 从训练文件夹中获取类别标签列表
def get_class_labels(train_folder):
    class_labels = []
    for class_name in os.listdir(train_folder):
        if os.path.isdir(os.path.join(train_folder, class_name)):
            class_labels.append(class_name)
    return class_labels

# 加载模型
def load_model(model_path, num_classes):
    model = AutoModelForImageClassification.from_pretrained(model_path, num_classes=num_classes)
    model.to(device)
    return model

# 图像预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    return image

# 图像分类
def classify_image(image_path, model, class_labels):
    image = preprocess_image(image_path)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.logits, 1)
        predicted_label = class_labels[predicted.item()]
    return predicted_label

# 示例用法
if __name__ == "__main__":
    train_folder = "dataset/cifar-10-batches/train"
    model_path = "/workspace/ImageClassification/checkpoints/beit-base-patch16-224_cifar-10-batchesbest_model.pth"  # 替换为微调后的模型路径
    num_classes = 10  # 替换为分类的类别数量
    class_labels = get_class_labels(train_folder)  # 替换为类别标签的列表
    image_path = "path/to/image.jpg"  # 替换为输入图像的路径

    # 加载模型
    model = load_model(model_path, num_classes)

    # 进行图像分类
    predicted_label = classify_image(image_path, model, class_labels)

    # 输出预测结果
    print("Predicted label:", predicted_label)
