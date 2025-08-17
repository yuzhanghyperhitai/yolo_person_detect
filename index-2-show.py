"""
检测人物矩形框，并实时显示。
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

def find_no_person_intervals(video_path, model_name='yolov8n.pt', confidence_threshold=0.5, show_video=True):
    """
    在视频中找出没有人物出现的时间片段，并实时显示检测过程。

    参数:
    video_path (str): 输入视频文件的路径
    model_name (str): YOLOv8模型的名称 (例如 'yolov8n.pt', 'yolov8s.pt')
    confidence_threshold (float): 人物检测的置信度阈值
    show_video (bool): 是否显示实时检测窗口
    """
    # 1. 加载模型，并指定在GPU上运行
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
    print(f"视频信息: FPS={fps:.2f}, 总帧数={total_frames}")

    # --- 新增：创建用于实时播放的窗口 ---
    if show_video:
        cv2.namedWindow("YOLOv8 Real-time Detection", cv2.WINDOW_NORMAL)

    # 3. 逐帧处理
    no_person_timestamps = []
    current_frame_idx = 0
    step =1   # 跳帧大于1时,疯狂加速
    while cap.isOpened():
        ret = cap.grab()
        if not ret:
            break
        if current_frame_idx % step == 0:
            retrieve_ret,frame= cap.retrieve()
            if not retrieve_ret:
                print(f"错误: 获取{ current_frame_idx }帧失败。")
                break            
            
            # 使用YOLOv8进行检测
            results = model(frame, classes=[0], conf=confidence_threshold, verbose=False)
            
            person_detected = False
            if results[0].boxes.shape[0] > 0:
                person_detected = True

            if not person_detected:
                current_time_sec = current_frame_idx / fps
                no_person_timestamps.append(current_time_sec)
            
            # --- 新增：实时绘制与播放 ---
            if show_video:
                # 使用YOLOv8内置的plot()函数，它会返回一个绘制好检测框的图像
                annotated_frame = results[0].plot()

                # 在左上角添加状态文本
                status = "Person Detected"
                color = (0, 0, 255) # 红色
                if not person_detected:
                    status = "No Person Detected"
                    color = (0, 255, 0) # 绿色
                
                cv2.putText(annotated_frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(annotated_frame, f"Frame: {current_frame_idx}/{total_frames}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


                # 显示处理后的帧
                cv2.imshow("YOLOv8 Real-time Detection", annotated_frame)

                # 按'q'键退出播放
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("用户提前退出播放。")
                    break
            # --- 结束新增部分 ---

            # 打印处理进度
            if current_frame_idx % int(fps * 5) == 0:
                print(f"已处理 {current_frame_idx} / {total_frames} 帧...")
        current_frame_idx += 1

    # --- 新增：释放资源 ---
    cap.release()
    if show_video:
        cv2.destroyAllWindows()
    print("视频处理完成。")

    # 4. 合并时间片段
    if not no_person_timestamps:
        print("视频中始终有人物存在（或未检测到无人片段）。")
        return

    print("正在合并时间片段...")
    intervals = []
    start_time = no_person_timestamps[0]
    end_time = no_person_timestamps[0]

    for i in range(1, len(no_person_timestamps)):
        if no_person_timestamps[i] - end_time < (1.0 / fps) * 1.5:
            end_time = no_person_timestamps[i]
        else:
            intervals.append((start_time, end_time))
            start_time = no_person_timestamps[i]
            end_time = no_person_timestamps[i]
    
    intervals.append((start_time, end_time))

    # 5. 输出结果
    print("\n--- 检测结果 ---")
    print("在以下时间片段内未检测到人物：")
    for start, end in intervals:
        print(f"从 {seconds_to_hms(start)} 到 {seconds_to_hms(end)}")

if __name__ == '__main__':
    # --- 配置区 ---
    # video_file = "/home/yuzhang/Hyperhit/dev-resource/1.裙子1min.mp4" 
    video_file = "/home/yuzhang/下载/wuyanzu/234a5e17-4faa-453e-b2ce-51c643b2f135.mp4"
    model_choice = 'yolov8x.pt'
    confidence = 0.5

    # 运行主函数，实时显示默认开启
    # 如果你只想分析不想看实时画面，可以将 show_video 改为 False
    find_no_person_intervals(
        video_file, 
        model_name=model_choice, 
        confidence_threshold=confidence,
        show_video=True 
    )