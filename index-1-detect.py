"""
检测视频中无角色的时间片段
"""

import cv2
from ultralytics import YOLO
import torch

def seconds_to_hms(seconds):
    """将秒数转换为时:分:秒.毫秒的格式"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def find_no_person_intervals(video_path, model_name='yolov8n.pt', confidence_threshold=0.5, step=10):
    """
    在视频中找出没有人物出现的时间片段

    参数:
    video_path (str): 输入视频文件的路径
    model_name (str): YOLOv8模型的名称 (例如 'yolov8n.pt', 'yolov8s.pt')
    confidence_threshold (float): 人物检测的置信度阈值
    """
    # 1. 加载模型，并指定在GPU上运行
    # device='cuda' or device=0
    print("正在加载YOLOv8模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"将使用设备: {device}")
    model = YOLO(model_name)
    model.to(device)

    # 2. 加载视频
    print(f"正在打开视频文件: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("错误: 无法打开视频文件。")
        return

    # 获取视频的基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    
    print(f"视频信息: FPS={fps:.2f}, 总帧数={total_frames}, 格式: {cap.get(cv2.CAP_PROP_FORMAT)}")

    # 3. 逐帧处理
    no_person_timestamps = []
    current_frame = 0
    while cap.isOpened():
        ret = cap.grab()      
        if not ret:
            break
        if current_frame % step == 0:
            retrieve_ret, frame = cap.retrieve()
            if not retrieve_ret:
                print(f"错误: 获取{ current_frame }帧失败。")
                break            
                
            # 使用YOLOv8进行检测
            # verbose=False可以禁止打印每一帧的检测日志
            results = model(frame, classes=[0], conf=confidence_threshold, verbose=False) # classes=[0] 表示只检测'person'类别
            person_detected = False
            # results[0].boxes.shape[0] > 0 表示检测到了目标
            if results[0].boxes.shape[0] > 0:
                person_detected = True

            if not person_detected:
                # 如果没有检测到人，记录当前时间戳（秒）
                current_time_sec = current_frame / fps
                no_person_timestamps.append(current_time_sec)
            
            # 打印处理进度
            print(f"已处理 {current_frame} / {total_frames} 帧...")                          
        current_frame += 1
    cap.release()
    print("视频处理完成。")

    # 4. 合并时间片段
    if not no_person_timestamps:
        print("视频中始终有人物存在。")
        return

    print("正在合并时间片段...", no_person_timestamps)
    intervals = []
    start_time = no_person_timestamps[0]
    end_time = no_person_timestamps[0]

    for i in range(1, len(no_person_timestamps)):
        # 允许 0.1s的误差
        if no_person_timestamps[i] - end_time < 0.5 :
            end_time = no_person_timestamps[i]
        else:
            intervals.append((start_time, end_time))
            start_time = no_person_timestamps[i]
            end_time = no_person_timestamps[i]
    
    intervals.append((start_time, end_time)) # 添加最后一个片段

    # 5. 输出结果
    print("\n--- 检测结果 ---")
    print("在以下时间片段内未检测到人物：")
    for start, end in intervals:
        print(f"从 {seconds_to_hms(start)} 到 {seconds_to_hms(end)}")


if __name__ == '__main__':
    # --- 配置区 ---
    # 将这里的路径换成你自己的MP4文件路径
    video_file = "/home/yuzhang/Hyperhit/dev-resource/1.裙子1min.mp4" 
    
    # 你可以选择不同的YOLOv8模型，n(nano)最快但精度稍低，s(small), m(medium), l(large), x(xlarge) 速度递减，精度递增
    # 对于RTX 4060，使用 'yolov8s.pt' 或 'yolov8m.pt' 是一个很好的平衡点
    model_choice = 'yolov8x.pt'
    
    # 置信度阈值，只有当模型认为某个物体是人的概率大于这个值时，才算检测到
    # 可以适当调整，比如0.4或0.6
    confidence = 0.5

    find_no_person_intervals(video_file, model_name=model_choice, confidence_threshold=confidence)