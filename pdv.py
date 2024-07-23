from ultralytics import YOLO
import cv2

# 加载训练后的模型
model = YOLO('train/weights/best.pt')

# 打开视频文件
video_path = 'test/test.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率和尺寸
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建 VideoWriter 对象A
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("无法读取视频帧")
        break
    
    # 使用YOLO模型进行预测
    results = model(frame)
    
    # 可视化结果
    annotated_frame = results[0].plot()
    
    # 写入处理后的帧到输出视频
    out.write(annotated_frame)
    
    # 显示结果
    cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    
    # 按'q'键退出
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()  # 释放 VideoWriter
cv2.destroyAllWindows()