from ultralytics import YOLO

# 加载一个模型
model = YOLO('yolov8n.yaml')  # 从YAML建立一个新模型
model = YOLO('yolov8n.pt')  # 加载预训练模型（推荐用于训练）
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # 从YAML建立并转移权重


if __name__ == "__main__":
    # 训练模型
    results = model.train(data='armor.yaml', epochs=10, imgsz=320, half=True)

    # 评估模型
    print("评估模型...")
    results = model.val()
    metrics = results.box.map  # 获取mAP@0.5:0.95
    metrics = results.box.map50  # 获取mAP@0.5

    print(metrics)