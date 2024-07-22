from ultralytics import YOLO
import cv2

# 加载训练后的模型
# model = YOLO('yolov8n.pt')
model = YOLO('runs/detect/train/weights/best.pt')

# 打开视频文件
# video_path = '/home/dbink/download/2021-0417-1559-小组赛-四川大学VS遵义师范-步兵1.mp4'
video_path = 'test/test.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    # 读取一帧
    success, frame = cap.read()
    
    if not success:
        print("无法读取视频帧")
        break
    
    # 使用YOLO模型进行预测
    results = model(frame)
    
    # 可视化结果
    annotated_frame = results[0].plot()
    
    # 显示结果
    cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    
    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()