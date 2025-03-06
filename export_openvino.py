from numpy import half
from ultralytics import YOLO

# Load the YOLOv11 model
model = YOLO("./320_11/320_11.pt")

# Export the model to RKNN format
# 'name' can be one of rk3588, rk3576, rk3566, rk3568
# model.export(format="rknn", name="rk3588")  # creates '/yolo11n_rknn_model'

# model.export(format="openvino", int8=True, imgsz=320)
# model.export(format="openvino", half=True, imgsz=320)
# model.export(format="openvino", imgsz=320)
