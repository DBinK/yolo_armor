import os
import cv2
import torch
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

