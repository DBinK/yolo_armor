import os
import cv2
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO

from torch.utils.data import Dataset, DataLoader

class ArmorDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.images[idx])
        image = cv2.imread(img_name)
        label_name = os.path.join(self.labels_dir, self.images[idx].replace('.jpg', '.txt'))
        with open(label_name, 'r') as f:
            labels = f.readlines()

        # 处理标签数据（根据需要进行解析）
        labels = [list(map(float, label.strip().split())) for label in labels]

        if self.transform:
            image = self.transform(image)

        return image, labels
    
# 使用数据集
images_dir = 'armor_dataset_v4/images'
labels_dir = 'armor_dataset_v4/labels'
dataset = ArmorDataset(images_dir, labels_dir)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 定义图像转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 加载模型
model = YOLO('runs/detect/train2/weights/best.pt')  # 替换为您的模型路径
model.eval()

# 准备量化
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# 校准模型
with torch.no_grad():
    for images, _ in data_loader:
        model(images)

# 转换为量化模型
quantized_model = torch.quantization.convert(model, inplace=True)

# 保存量化模型
torch.save(quantized_model.state_dict(), 'quantized_yolov8.pt')