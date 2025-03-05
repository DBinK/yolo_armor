import cv2
import time
from torch import device
from ultralytics import YOLO

# 加载 YOLO 模型
# model = YOLO("best.pt")  
# model.to(device='cpu')

# model = YOLO("./640/640.pt")  
# model.to(device='cpu')
# model = YOLO("D:\\IT\\yolo_armor\\640\\640_openvino_model")  
# model = YOLO("D:\\IT\\yolo_armor\\640\\640_half_openvino_model")  
# model = YOLO("D:\\IT\\yolo_armor\\640\\640_int8_openvino_model")  

# model = YOLO("./320/320.pt")  
# model.to(device='cpu')
model = YOLO("D:\\IT\\yolo_armor\\320\\320_openvino_model")  
# model = YOLO("D:\\IT\\yolo_armor\\320\\320_half_openvino_model")  
# model = YOLO("D:\\IT\\yolo_armor\\320\\320_int8_openvino_model")  

# model = YOLO("D:\\IT\\yolo_armor\\runs\\detect\\train12\\weights\\best.pt")  

video_path = "test/test3.mp4"  # 替换为你的视频路径
# video_path = 0  # 实时摄像头推理

cap = cv2.VideoCapture(video_path)


# 初始化性能指标
inference_times = []
frame_count = 0

# 初始化 待识别目标 计数器
tg_name = "B3"
tg_count = 0

# 初始化类别计数器
class_counts = {}

# 设置忽略的帧数, 等待推理帧率稳定
skip_frames = 50  

# 读取视频帧并进行推理
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1

    # # 跳过前 N 帧
    if frame_count <= skip_frames:
        print(f"跳过前 {frame_count} 帧...")

    # # 跳过前 N 帧
    # if frame_count > skip_frames:
    #     start_time = time.time()  # 开始计时

    # # 跳过前 N 帧
    if frame_count == 50:
        start_time = time.time()  # 开始计时
    
    # 推理
    # results = model(frame)
    results = model.predict(frame, save=False, imgsz=320)
    
    # # 获取检测结果
    boxes = results[0].boxes.cpu().numpy()

    # # 遍历检测结果
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        
        # 更新类别计数器
        if cls_name in class_counts:
            class_counts[cls_name] += 1
        else:
            class_counts[cls_name] = 1
        
        if cls_name == tg_name:
            tg_count += 1

    # # 可视化结果
    # annotated_frame = results[0].plot()
    # cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
    # cv2.imshow("YOLOv8 Inference", annotated_frame)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break            

    # 跳过前 N 帧
    # if frame_count > skip_frames:
    #     end_time = time.time()  # 结束计时
    #     inference_time = end_time - start_time  # 记录推理时间
    #     inference_times.append(inference_time)
    #     print(f"帧 {frame_count}: 推理时间 = {inference_time*1000:.4f} 秒")

end_time = time.time()  # 结束计时
spend_time = end_time - start_time  # 记录总推理时间

average_inference_time = spend_time / (frame_count-skip_frames)
fps = 1 / average_inference_time

# 释放视频捕获
cap.release()

# 计算性能指标
# average_inference_time = sum(inference_times) / len(inference_times)
# max_inference_time = max(inference_times)
# min_inference_time = min(inference_times)
# fps = frame_count / sum(inference_times)

# 打印结果
print(f"处理帧数: {frame_count}")
print(f"平均推理时间: {average_inference_time:.4f} 秒")
# print(f"瞬时最大推理时间: {max_inference_time:.4f} 秒")
# print(f"瞬时最小推理时间: {min_inference_time:.4f} 秒")
print(f"平均帧率: {fps:.2f} FPS")
print(f"识别到 B3 的次数: {tg_count}")

# 打印每个类别的计数
print("每个类别的计数:")
for cls_name, count in class_counts.items():
    print(f"{cls_name}: {count}")