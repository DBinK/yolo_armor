import os
import shutil
import random

def copy_files(src_dir, dst_dir, file_list):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for file_name in file_list:
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(dst_dir, file_name)
        shutil.copy(src_file, dst_file)

def create_new_dataset(source_dir, target_dir, num_samples=1000):
    images_dir = os.path.join(source_dir, 'images')
    labels_dir = os.path.join(source_dir, 'labels')
    
    # 获取所有图像文件名
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 随机选择num_samples个图像文件
    selected_images = random.sample(image_files, num_samples)
    
    # 提取对应的标签文件名
    selected_labels = [os.path.splitext(f)[0] + '.txt' for f in selected_images]
    
    # 创建目标目录
    new_images_dir = os.path.join(target_dir, 'images')
    new_labels_dir = os.path.join(target_dir, 'labels')
    
    # 复制选定的图像和标签文件
    copy_files(images_dir, new_images_dir, selected_images)
    copy_files(labels_dir, new_labels_dir, selected_labels)
    
    print(f"成功创建新的数据集，包含 {num_samples} 张图像及其标签。")

if __name__ == "__main__":
    source_dataset_path = 'armor_dataset_v4'  # 原始数据集路径
    target_dataset_path = 'new_dataset'  # 新数据集路径
    num_samples = 1000  # 要抽取的样本数量
    
    create_new_dataset(source_dataset_path, target_dataset_path, num_samples)