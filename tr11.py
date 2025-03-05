from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
# results = model.train(data="coco8.yaml", epochs=5, imgsz=640, half=True)

if __name__ == "__main__":
    # 训练模型
    results = model.train(data='armor.yaml', epochs=10, imgsz=320, half=True)

    # 评估模型
    results = model.val()
    metrics = results.box.map  # 获取mAP@0.5:0.95
    metrics = results.box.map50  # 获取mAP@0.5

    print(metrics)

    # model.export(format="rknn", name="rk3588", )

    # model.export(format="rknn", name="rk3568")


