import cv2
import apriltag
import numpy as np
import os
import sys

def is_usb_camera(device):
    try:
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            return False
        cap.release()
        return True
    except Exception:
        return False

def find_first_usb_camera():
    video_devices = [os.path.join('/dev', dev) for dev in os.listdir('/dev') if dev.startswith('video')]
    for dev in video_devices:
        if is_usb_camera(dev):
            return dev
    return None

def detect_apriltag_realtime():
    """实时检测AprilTag"""
    
    if len(sys.argv) > 1:
        video_device = sys.argv[1]
    else:
        video_device = find_first_usb_camera()

    if video_device is None:
        print("No USB camera found.")
        sys.exit(-1)

    print(f"Opening video device: {video_device}")
    cap = cv2.VideoCapture(video_device)
    if(not cap.isOpened()):
        exit(-1)
    
    print("Open usb camera successfully")
    # 设置usb camera的输出图像格式为 MJPEG， 分辨率 640 x 480
    # 可以通过 v4l2-ctl -d /dev/video8 --list-formats-ext 命令查看摄像头支持的分辨率
    # 根据应用需求调整该采集图像的分辨率
    codec = cv2.VideoWriter_fourcc( 'M', 'J', 'P', 'G' )
    cap.set(cv2.CAP_PROP_FOURCC, codec)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    
    # 创建AprilTag检测器
    options = apriltag.DetectorOptions(
        # 指定标签族
        families="tag25h9",
        
        # 边缘检测参数（关键）
        quad_decimate=1.0,       # 降采样因子，减小可提高精度但降低速度（默认1.0）
        # quad_sigma=0.8,          # 高斯模糊系数，增加可减少噪声（默认0.0表示不模糊）
        refine_edges=True,       # 优化边缘检测（默认True）
        
        # 解码参数
        # decode_sharpening=0.25,  # 增强图像锐度以提高解码成功率（默认0.25）
        
        # 阈值参数（关键）
        # min_white_black_diff=50, # 最小黑白对比度（默认20，增加可减少误检）
        # quad_min_area=1000,      # 最小检测区域（像素），过滤小区域
        # max_line_fit_mse=10.0,   # 最大线段拟合误差，增加可容忍更多变形
        # quad_max_nmaxima=10,     # 最大轮廓顶点数，减少可过滤复杂形状
        # quad_max_error=2.0,      # 四边形拟合最大误差（默认3.0，减小可提高精度）
        
        # 多线程加速
        nthreads=4,
    )
    detector = apriltag.Detector(options)

    while True:
        # 读取一帧视频
        ret ,frame = cap.read()
            
        # if frame is None:
        #     print("Failed to get image from usb camera")
        #     break
        
        # 转换为灰度图像（AprilTag检测需要）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测AprilTag
        results = detector.detect(gray)
        
        # 在原始彩色图像上绘制检测结果
        for result in results:
            # 提取标签的角点
            corners = result.corners.astype(int)
            
            if len(result.corners) < 4:
                print(f"警告：标签ID {result.tag_id} 的角点数量不足")
                continue

            # 绘制边框
            for i in range(4):
                cv2.line(frame, tuple(corners[i]), tuple(corners[(i+1)%4]), (0, 255, 0), 2)
            
            # 绘制标签ID
            tag_id = result.tag_id
            cv2.putText(frame, f"ID: {tag_id}", 
                        (int(result.center[0]), int(result.center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 打印标签信息
            print(f"检测到AprilTag ID: {tag_id}, 中心位置: {result.center}")
        
        # 显示结果
        cv2.imshow("AprilTag", frame)
        
        # 按 'q' 键退出循环
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC键或q键
            break
        
        # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    
    detect_apriltag_realtime()    