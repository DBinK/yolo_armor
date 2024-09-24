from ultralytics import YOLO
import cv2
import time
import numpy as np
from datetime import datetime  # 导入datetime模块以获取当前时间

# 加载训练后的模型
model_path = 'train4/weights/best.pt'
model = YOLO(model_path)

# 打开视频文件
video_path = 'test/test.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率和尺寸
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建 VideoWriter 对象
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 用于记录延迟的列表
latencies = []

# 记录开始时间
start_test_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("无法读取视频帧")
        break

    # 记录处理开始时间
    frame_start_time = time.time()
    
    # 使用YOLO模型进行预测
    results = model(frame)
    
    # 记录处理结束时间
    frame_end_time = time.time()
    
    # 可视化结果
    annotated_frame = results[0].plot()
    
    # 写入处理后的帧到输出视频
    out.write(annotated_frame)
    
    # 显示结果
    # cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
    # cv2.imshow("YOLOv8 Inference", annotated_frame)
    
    # 按'q'键退出
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

    # 计算延迟
    latency = frame_end_time - frame_start_time
    latencies.append(latency)  # 记录每一帧的延迟

# 释放资源
cap.release()
out.release()  # 释放 VideoWriter
cv2.destroyAllWindows()

# 记录结束时间
end_test_time = time.time()
total_test_time = end_test_time - start_test_time  # 总测试时间

# 计算最大、最小、平均延迟和标准差
if latencies:
    max_latency = max(latencies)  # 最大延迟
    min_latency = min(latencies)  # 最小延迟
    avg_latency = np.mean(latencies)  # 平均延迟
    std_dev_latency = np.std(latencies)  # 延迟标准差
    
    print(f"总测试时间: {total_test_time:.4f}秒")
    print(f"最大延迟: {1000*max_latency:.4f} ms")
    print(f"最小延迟: {1000*min_latency:.4f} ms")
    print(f"平均延迟: {1000*avg_latency:.4f} ms")
    print(f"延迟标准差: {1000*std_dev_latency:.4f} ms")


    # 生成报告
    report_filename = datetime.now().strftime("%Y-%m-%d_%H-%M.md")  # 生成文件名

    with open(report_filename, 'w', encoding='utf-8') as report_file:

        report_file.write("# YOLOv8 测试报告\n")
        report_file.write(f"\n模型文件路径: {model_path}\n")
        report_file.write(f"总测试时间: {total_test_time:.4f}秒\n")
        report_file.write(f"最大延迟: {1000*max_latency:.4f} ms\n")
        report_file.write(f"最小延迟: {1000*min_latency:.4f} ms\n")
        report_file.write(f"\n平均延迟: {1000*avg_latency:.4f} ms\n")
        report_file.write(f"\n延迟标准差: {1000*std_dev_latency:.4f} ms\n")

    print(f"报告已生成: {report_filename}")