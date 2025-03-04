from ultralytics import YOLO
import cv2

# 加载训练后的模型
model = YOLO('train/weights/best.pt')
# model = YOLO('D:\\IT\\yolo_armor\\runs\\detect\\train11\\weights\\best.pt')

# 打开视频文件
# video_path = 'test.mp4'
video_path = 0
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)


while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("无法读取视频帧")
        break
    
    # 使用YOLO模型进行预测
    results = model(frame)
    
    # 可视化结果
    annotated_frame = results[0].plot()
    # annotated_frame = frame
    
    # 显示结果
    cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    
    # 根据帧率计算延迟时间，确保视频以原速度播放
    delay_time = max(1, int(1000 / fps))
    if cv2.waitKey(delay_time) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()