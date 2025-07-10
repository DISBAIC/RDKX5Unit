#!/user/bin/env python

# Copyright (c) 2024，WuChao D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 注意: 此程序在RDK板端端运行
# Attention: This program runs on RDK board.
import serial
import cv2
import numpy as np
from scipy.special import softmax
# from scipy.special import expit as sigmoid
from hobot_dnn import pyeasy_dnn as dnn  # BSP Python API
import sys
from time import time
import argparse
import logging
import ctypes
import os
import math
from zoom1 import process

from first_bsp_realtime import BPU_Detect
from serial1 import sending_data
from path_finder import search_path

# 日志模块配置
# logging configs
logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")


print(sys.executable)

def signal_handler(signal, frame):
    print("\nExiting program")
    sys.exit(0)

def limit_display_cord(coor):
    coor[0] = max(min(1920, coor[0]), 0)
    # min coor is set to 2 not 0, leaving room for string display
    coor[1] = max(min(1080, coor[1]), 2)
    coor[2] = max(min(1920, coor[2]), 0)
    coor[3] = max(min(1080, coor[3]), 0)
    return coor

libpostprocess = ctypes.CDLL('/usr/lib/libpostprocess.so')

def get_display_res():
    if os.path.exists("/usr/bin/get_hdmi_res") == False:
        return 1920, 1080

    import subprocess
    p = subprocess.Popen(["/usr/bin/get_hdmi_res"], stdout=subprocess.PIPE)
    result = p.communicate()
    res = result[0].split(b',')
    res[1] = max(min(int(res[1]), 1920), 0)
    res[0] = max(min(int(res[0]), 1080), 0)
    return int(res[1]), int(res[0])

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

def print_properties(pro):
    print("tensor type:", pro.tensor_type)
    print("data type:", pro.dtype)
    print("layout:", pro.layout)
    print("shape:", pro.shape)

def pixelate_mask(mask, target_size=100, keep_aspect_ratio=True):
    """
    对掩码进行像素化处理，保持原始宽高比（不裁剪为正方形），缩小像素尺寸
    
    参数:
    mask: 原始掩码图 (numpy数组)
    target_size: 目标短边尺寸（控制缩放程度，短边会缩放到此值）
    keep_aspect_ratio: 是否保持原始宽高比（固定为True，确保比例不变）
    
    返回:
    像素化后的掩码图 (numpy数组，保持原始宽高比，短边为target_size)
    """
    # 确保输入是二维或三维数组
    if len(mask.shape) == 2:
        h, w = mask.shape
        is_gray = True
    elif len(mask.shape) == 3:
        h, w, c = mask.shape
        is_gray = c == 1
    else:
        raise ValueError("输入掩码必须是二维或三维数组")
    
    # 强制保持宽高比（忽略传入的False，确保比例不变）
    keep_aspect_ratio = True  # 固定为True，确保不拉伸图像
    
    if keep_aspect_ratio:
        # 保持原始宽高比，将短边缩放到target_size（核心修改：不裁剪为正方形）
        scale = target_size / min(h, w)
        target_h, target_w = int(h * scale), int(w * scale)
        
        # 确保目标尺寸至少为1像素
        target_h = max(1, target_h)
        target_w = max(1, target_w)
        
        # 第一步：降采样到目标尺寸（生成像素化效果）
        # 直接缩放到目标尺寸（短边=target_size，长边按比例缩放）
        pixelated = cv2.resize(
            mask, 
            (target_w, target_h), 
            interpolation=cv2.INTER_NEAREST  # 最近邻插值，保持硬边缘
        )
        
        # 核心修改：移除裁剪和填充正方形的逻辑，直接保留按比例缩放的结果
        result = pixelated
    
    else:
        # 即使传入False，仍按保持比例处理（避免拉伸，与需求一致）
        scale = target_size / min(h, w)
        target_h, target_w = int(h * scale), int(w * scale)
        target_h = max(1, target_h)
        target_w = max(1, target_w)
        result = cv2.resize(
            mask, 
            (target_w, target_h), 
            interpolation=cv2.INTER_NEAREST
        )
    
    return result.astype(np.uint8)

target_colors = [(31, 112, 255), (29, 178, 255),(199, 55, 255)]

def binarize_image_to_array(image, target_colors, tolerance=30, expand_size=2):
    """
    将图片二值化并转换为二维数组，同时扩大目标颜色区域
    
    参数:
    image (str/numpy.ndarray): 图片路径或图片数组
    target_colors (list): 目标颜色列表，每个颜色格式为(B,G,R)
    tolerance (int): 颜色匹配容忍度，值越大匹配范围越广
    expand_size (int): 目标区域向外扩展的像素数
    
    返回:
    numpy.ndarray: 二值化后的二维数组，黑色为0，白色为1
    """
    # 判断输入是路径还是数组
    if isinstance(image, str):
        # 如果是路径，则读取图片
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"无法读取图片: {image}")
    else:
        # 如果是数组，直接使用
        img = image
    # if img.size == 0:
    #     raise ValueError("图像数据为空，请检查输入图像是否有效")
    # 创建二值化掩码
    # 黑色区域
    # if img != None:
    black_mask = cv2.inRange(img, (0, 0, 0), (50, 50, 50))
    
    # 初始化颜色掩码
    color_mask = np.zeros_like(black_mask)
    
    # 为每种目标颜色创建掩码并合并
    for color in target_colors:
        lower_bound = np.array([max(0, c - tolerance) for c in color])
        upper_bound = np.array([min(255, c + tolerance) for c in color])
        color_mask = cv2.bitwise_or(color_mask, cv2.inRange(img, lower_bound, upper_bound))
    
    # 合并所有掩码（黑色和所有目标颜色都变为黑色(0)）
    combined_mask = cv2.bitwise_or(black_mask, color_mask)
    
    # 扩大黑色区域（目标颜色及其周边）
    if expand_size > 0:
        # 创建膨胀核（可以是矩形、椭圆等）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_size*2+1, expand_size*2+1))
        # 应用膨胀操作
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    
    # 将合并后的掩码转换为0和1的二维数组
    # 黑色区域(0)保持为0，其他区域(255)变为1
    binary_array = (combined_mask == 0).astype(np.uint8).tolist()
    
    return binary_array

def main_map():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='/app/my_project/bin/converted_model_modified5.bin', 
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""") 
    # parser.add_argument('--test-img', type=str, default='../../../../resource/datasets/COCO2017/assets/bus.jpg', help='Path to Load Test Image.')
    # parser.add_argument('--test-img', type=str, default='/app/my_project/v8seg-photo/微信图片_20250615124202.jpg', help='Path to Load Test Image.')
    # parser.add_argument('--mask-save-path', type=str, default='/app/my_project/v8seg-photo/mask.jpg', help='Path to Save Mask Image.')
    # parser.add_argument('--img-save-path', type=str, default='/app/my_project/v8seg-photo/result.jpg', help='Path to Load Test Image.')
    parser.add_argument('--classes-num', type=int, default=5, help='Classes Num to Detect.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='IoU threshold.')
    parser.add_argument('--score-thres', type=float, default=0.8, help='confidence threshold.')
    parser.add_argument('--reg', type=int, default=16, help='DFL reg layer.')
    parser.add_argument('--mc', type=int, default=32, help='Mask Coefficients')
    parser.add_argument('--is-open', type=bool, default=True, help='Ture: morphologyEx')
    parser.add_argument('--is-point', type=bool, default=True, help='Ture: Draw edge points')
    opt = parser.parse_args()
    logger.info(opt)
    i=0
    Buildmap=0
    # 实例化
    model = YOLO11_Seg(opt)

    coconame1 = ["fire","smoke"]
    models1 = "/app/my_project/bin/converted_model2.bin"
    infer = BPU_Detect(models1,coconame1,conf=0.6,iou=0.3,mode = True)

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

    ser = serial.Serial('/dev/ttyS1', 115200, timeout=1) # 1s timeout
    t2=0
    car_center=(0,0)
    car_radius=0
    fire_center=(0,0)

    while True:
        # ser.write('R'.encode('UTF-8'))
        # if ser.readall():
        #     ser.write('R'.encode('UTF-8'))
        #     print("Success to connect to the serial device.")
            # aaaaaa=ser.readall()
            

        # 读取摄像头图像
        _ ,frame = cap.read()
            
        # print(frame.shape)
        if frame is None:
            print("Failed to get image from usb camera")
            continue

        des_dim = (768, 576)

        # 读图
        img = cv2.resize(frame, des_dim, interpolation=cv2.INTER_AREA)

        infer.detect(img)
        for class_id, score, bbox in zip(infer.ids, infer.scores, infer.bboxes):
            x11, y11, x22, y22 = bbox

        # 读图
        # 准备输入数据
        input_tensor = model.preprocess_yuv420sp(img)
        # 推理
        outputs = model.c2numpy(model.forward(input_tensor))
        # 后处理
        results = model.postProcess(outputs)
        # 渲染
        logger.info("\033[1;32m" + "Draw Results: " + "\033[0m")
        # 绘制
        draw_img = img.copy()
        zeros = np.zeros((img.shape[0],img.shape[1],3), dtype=np.uint8)
        # 类别计数
        class_counters = {}
        #掩码处理部分
        for class_id, score, x1, y1, x2, y2, mask in results:                
            # 计算车的实时方向
            if class_id == 3:
                tou = ((x1 + x2) // 2, (y1 + y2) // 2)
            if class_id == 4:
                wei = ((x1 + x2) // 2, (y1 + y2) // 2)

            try:
                dx = tou[0]-wei[0]
                dy = tou[1]-wei[1]
                #提取半径
                car_center=((tou[0]+wei[0])//2,(tou[1]+wei[1])//2)
                car_radius=abs(dy-dx)*0.75

                # 计算弧度角，注意：atan2 参数是 (y, x)
                angle_rad = math.atan2(dy, dx)

                # 转换为角度，并调整范围到 [0, 360)
                angle_deg = 180-math.degrees(angle_rad)
                if angle_deg < 0:
                    angle_deg += 360
                if angle_deg > 360:
                    angle_deg -= 360
                print(f"Direction Angle: {angle_deg:.2f}°")
            except NameError as e:
                print("缺少车头或车尾位置信息，无法计算方向。")

            print("(%d, %d, %d, %d) -> %s: %.2f"%(x1,y1,x2,y2, coco_names[class_id], score))
            draw_detection(draw_img, (x1, y1, x2, y2), score, class_id)
            # Instance Segment
            if mask.size == 0:
                continue
            mask = cv2.resize(mask, (int(x2-x1), int(y2-y1)), interpolation=cv2.INTER_LINEAR)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, model.kernel_for_morphologyEx, 1) if opt.is_open else mask      
            zeros[y1:y2,x1:x2, :][mask == 1] = rdk_colors[(class_id-1)%20]
            # points
            if not opt.is_point:
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # 手动连接轮廓
                contour = np.vstack((contours[0], np.array([contours[0][0]])))
                for i in range(1, len(contours)):
                    contour = np.vstack((contour, contours[i], np.array([contours[i][0]])))
                # 轮廓投射回原来的图像大小
                merged_points = contour[:,0,:]
                merged_points[:,0] = merged_points[:,0] + x1
                merged_points[:,1] = merged_points[:,1] + y1
                points = np.array([[[int(x), int(y)] for x, y in merged_points]], dtype=np.int32)
                # 绘制轮廓
                cv2.polylines(draw_img, points, isClosed=True, color=rdk_colors[(class_id-1)%20], thickness=4)
             
            # 更新类别计数器
            if class_id in class_counters:
                class_counters[class_id] += 1
            else:
                class_counters[class_id] = 1
            # if class_counters[1] == 7 and class_counters[2] == 4 and class_counters[3] == 1 and class_counters[4] == 1 and len(infer.ids) > 0:
            #     Buildmap=1
            # else:
            #     Buildmap=0    
        #把火焰方框添加到掩码图中
        fire_center=(0,0)
        fire_radius=0        
        if len(infer.ids) != 0: 
            # 获取rdk_colors的最后一个颜色
            last_color = rdk_colors[-1]
    
            for class_id, score, bbox in zip(infer.ids, infer.scores, infer.bboxes):
                x11, y11, x22, y22 = bbox
                fire_radius = math.sqrt((x11-x22)**2+(y11-y22)**2)*0.5
                fire_center = ((x11+x22)//2,(y11+y22)//2)
                # 绘制边界框
                if class_id == 0:
                    cv2.rectangle(draw_img, (x11, y11), (x22, y22), last_color, 2)
                    infer.draw_detection(draw_img, (x11, y11, x22, y22), score, class_id, infer.labelname)
                    # 在zeros掩码图中填充相同颜色（实心矩形）
                    zeros[y11:y22, x11:x22, :] = last_color

        add_result = np.clip(draw_img + 0.3*zeros, 0, 255).astype(np.uint8)
        #融合图像显示
        cv2.imshow("add_result",add_result)
        #透视变换
        zeros=process(zeros)
        #缩小地图与寻路
        
        t2 += 1
        print(t2)
        if t2>15:
            end_y=0
            end_x=0
            start_x, start_y = car_center[0], car_center[1]
            
            #画起点
            cv2.circle(zeros, (int(start_x), int(start_y)), 5, (0, 255, 0), -1)
            t0 = time()
            Zoom=des_dim[0]/133
            print(Zoom)
            print(int(car_center[0]/Zoom),int(car_center[1]/Zoom))
            # 低像素化
            map2 = pixelate_mask(zeros, 100)
            # 二值化
            map2=binarize_image_to_array(map2,[(31, 112, 255), (29, 178, 255),(199, 55, 255)], tolerance=1, expand_size=0)
    
            # result = search_path(map2,(int(car_center[0]/Zoom),int(car_center[1]/Zoom)),(int(fire_center[0]/Zoom),int(fire_center[1]/Zoom)),int(car_radius/Zoom),int(fire_radius/Zoom))
            result = search_path(map2,(int(car_center[0]/Zoom),int(car_center[1]/Zoom)),(10,120),int(car_radius/Zoom),int(fire_radius/Zoom))

            cv2.circle(zeros, (int(start_x), int(start_y)), 15, (0, 255, 0), -1)
            all_data=[]
            for r in result:
                print(r.angle,r.distance)

                target_angle=abs(int(angle_deg-r.angle))
                if target_angle==360:
                    target_angle=0
                distance=int(r.distance*26.15)#16.5
                #如果车的角度比要前进的角度大，则先负转弯，再前进
                if angle_deg > r.angle:
                    # sending_data(0x54,2,target_angle/255,target_angle%255,0x45)
                    # sending_data(0x54,0,distance/255,distance%255,0x45)
                    packet1 = bytes([0x54, 1, int(target_angle/255.0) % 255, target_angle%255, 0x45])
                    packet2 = bytes([0x54, 0, int(distance/255) % 255, distance%255, 0x45])
                    all_data.append(packet1)
                    all_data.append(packet2)
                    angle_deg=angle_deg-target_angle
                    if angle_deg < 0:
                        angle_deg += 360
                #如果车的角度比要前进的角度小，则先正转弯，再前进    
                else :
                    # sending_data(0x54,1,target_angle/255,target_angle%255,0x45)
                    # sending_data(0x54,0,distance/255,distance%255,0x45)
                    packet1 = bytes([0x54, 2, int(target_angle/255.0) % 255, target_angle%255, 0x45])
                    packet2 = bytes([0x54, 0, int(distance/255) % 255, distance%255, 0x45])
                    all_data.append(packet1) 
                    all_data.append(packet2)
                    angle_deg=angle_deg+target_angle
                    if angle_deg >= 360:
                        angle_deg -= 360

                angle_rad = math.radians(angle_deg)
                end_x = start_x + distance * math.cos(angle_rad)/26.15*Zoom#前是实际与低像素图比例，后是图片像素比
                end_y = start_y - distance * math.sin(angle_rad)/26.15*Zoom
                #画路径
                cv2.line(zeros, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0, 255, 0), 2)
                start_x, start_y = end_x, end_y

            ser.write(b''.join(all_data))

            print(all_data)
            # 画终点
            cv2.circle(zeros, (int(start_x), int(start_y)), 15, (255, 0, 0), -1)
            # 画火焰点

            # cv2.circle(zeros, fire_center, 15, (0, 0, 255), -1)
            t1 = time()
            print("forward time is :", (t1 - t0))
        
        #掩码显示
        cv2.imshow("Mask",zeros)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC键或q键
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    # # 可视化, 这里采用直接相加的方式，实际应用中对Mask一般不需要Resize这些操作
    # add_result = np.clip(draw_img + 0.3*zeros, 0, 255).astype(np.uint8)
    # # 保存结果
    # cv2.imwrite(opt.img_save_path, np.hstack((draw_img, zeros, add_result)))
    # # 单独保存掩码图
    # cv2.imwrite(opt.mask_save_path, zeros)
    # logger.info("\033[1;32m" + f"saved combined result in path: \"./{opt.img_save_path}\"" + "\033[0m")
    # logger.info("\033[1;32m" + f"saved mask result in path: \"./{opt.mask_save_path}\"" + "\033[0m")
    # # 在main()函数中，后处理完成后添加以下代码
    # class_mask = generate_class_mask(results, img.shape)

    # # 保存为图像（0-背景，1-第一类，2-第二类，依此类推）
    # # cv2.imwrite("class_mask.png", class_mask)

    # # 保存为numpy数组，便于后续处理
    # np.save("class_mask.npy", class_mask)

    # # 保存为文本文件
    # txt_save_path = "/app/my_project/v8seg-photo/class_mask.txt"  # 修改为你想保存的路径
    # np.savetxt(txt_save_path, class_mask, fmt='%d')  # 使用%d格式保存整数

    # # 打印统计信息
    # print(f"生成的类别掩码形状: {class_mask.shape}")
    # print(f"检测到的类别数量: {len(np.unique(class_mask)) - 1}")  # 减去背景(0)
    # print(f"文本文件已保存至: {txt_save_path}")
        


class YOLO11_Seg():
    def __init__(self, opt):
        # 加载BPU的bin模型, 打印相关参数
        # Load the quantized *.bin model and print its parameters
        try:
            begin_time = time()
            self.quantize_model = dnn.load(opt.model_path)
            logger.debug("\033[1;31m" + "Load D-Robotics Quantize model time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")
        except Exception as e:
            logger.error("❌ Failed to load model file: %s"%(opt.model_path))
            logger.error("You can download the model file from the following docs: ./models/download.md") 
            logger.error(e)
            exit(1)

        logger.info("\033[1;32m" + "-> input tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].inputs):
            logger.info(f"intput[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        logger.info("\033[1;32m" + "-> output tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].outputs):
            logger.info(f"output[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        # 将反量化系数准备好, 只需要准备一次
        # prepare the quantize scale, just need to generate once
        self.s_bboxes_scale = self.quantize_model[0].outputs[1].properties.scale_data[np.newaxis, :]
        self.m_bboxes_scale = self.quantize_model[0].outputs[4].properties.scale_data[np.newaxis, :]
        self.l_bboxes_scale = self.quantize_model[0].outputs[7].properties.scale_data[np.newaxis, :]
        logger.info(f"{self.s_bboxes_scale.shape=}, {self.m_bboxes_scale.shape=}, {self.l_bboxes_scale.shape=}")

        self.s_mces_scale = self.quantize_model[0].outputs[2].properties.scale_data[np.newaxis, :]
        self.m_mces_scale = self.quantize_model[0].outputs[5].properties.scale_data[np.newaxis, :]
        self.l_mces_scale = self.quantize_model[0].outputs[8].properties.scale_data[np.newaxis, :]
        logger.info(f"{self.s_mces_scale.shape=}, {self.m_mces_scale.shape=}, {self.l_mces_scale.shape=}")

        self.mask_scale = self.quantize_model[0].outputs[9].properties.scale_data[0]
        logger.info(f"{self.mask_scale = }")


        # DFL求期望的系数, 只需要生成一次
        # DFL calculates the expected coefficients, which only needs to be generated once.
        self.weights_static = np.array([i for i in range(16)]).astype(np.float32)[np.newaxis, np.newaxis, :]
        logger.info(f"{self.weights_static.shape = }")

        # anchors, 只需要生成一次
        # self.s_anchor = np.stack([np.tile(np.linspace(0.5, 79.5, 80), reps=80), 
        #                     np.repeat(np.arange(0.5, 80.5, 1), 80)], axis=0).transpose(1,0)
        # self.m_anchor = np.stack([np.tile(np.linspace(0.5, 39.5, 40), reps=40), 
        #                     np.repeat(np.arange(0.5, 40.5, 1), 40)], axis=0).transpose(1,0)
        # self.l_anchor = np.stack([np.tile(np.linspace(0.5, 19.5, 20), reps=20), 
        #                     np.repeat(np.arange(0.5, 20.5, 1), 20)], axis=0).transpose(1,0)

        # 小目标检测层（s_anchor）：原尺寸 80x80 -> 新尺寸 96x72
        self.s_anchor = np.stack([np.tile(np.linspace(0.5, 95.5, 96), reps=72),
                          np.repeat(np.arange(0.5, 72.5, 1), 96)], axis=0).transpose(1, 0)
        # 中目标检测层（m_anchor）：原尺寸 40x40 -> 新尺寸 48x36
        self.m_anchor = np.stack([np.tile(np.linspace(0.5, 47.5, 48), reps=36),
                          np.repeat(np.arange(0.5, 36.5, 1), 48)], axis=0).transpose(1, 0)
        # 大目标检测层（l_anchor）：原尺寸 20x20 -> 新尺寸 24x18
        self.l_anchor = np.stack([np.tile(np.linspace(0.5, 23.5, 24), reps=18),
                          np.repeat(np.arange(0.5, 18.5, 1), 24)], axis=0).transpose(1, 0)

        logger.info(f"{self.s_anchor.shape = }, {self.m_anchor.shape = }, {self.l_anchor.shape = }")

        # 输入图像大小, 一些阈值, 提前计算好
        self.input_image_size = (576, 768)
        self.SCORE_THRESHOLD = opt.score_thres
        self.NMS_THRESHOLD = opt.nms_thres
        self.CONF_THRES_RAW = -np.log(1/self.SCORE_THRESHOLD - 1)
        logger.info("SCORE_THRESHOLD  = %.2f, NMS_THRESHOLD = %.2f"%(self.SCORE_THRESHOLD, self.NMS_THRESHOLD))
        logger.info("CONF_THRES_RAW = %.2f"%self.CONF_THRES_RAW)

        self.input_H, self.input_W = self.quantize_model[0].inputs[0].properties.shape[2:4]
        logger.info(f"{self.input_H = }, {self.input_W = }")

        self.Mask_H, self.Mask_W = 144, 192
        self.x_scale_corp = self.Mask_W / self.input_W
        self.y_scale_corp = self.Mask_H / self.input_H
        logger.info(f"{self.Mask_H = }   {self.Mask_W = }")
        logger.info(f"{self.x_scale_corp = }, {self.y_scale_corp = }")

        self.REG = opt.reg
        logger.info(f"{self.REG = }")

        self.CLASSES_NUM = opt.classes_num
        logger.info(f"{self.CLASSES_NUM = }")

        self.MCES_NUM = opt.mc
        logger.info(f"{self.MCES_NUM = }")

        self.IS_OPEN = opt.is_open  # 是否对Mask进行形态学开运算
        self.kernel_for_morphologyEx = np.ones((5,5), np.uint8) 
        logger.info(f"{self.IS_OPEN = }   {self.kernel_for_morphologyEx = }")

        self.IS_POINT = opt.is_point # 是否绘制边缘点
        logger.info(f"{self.IS_POINT = }")

    def preprocess_yuv420sp(self, img):
        RESIZE_TYPE = 0
        LETTERBOX_TYPE = 1
        PREPROCESS_TYPE = LETTERBOX_TYPE
        logger.info(f"PREPROCESS_TYPE = {PREPROCESS_TYPE}")

        begin_time = time()
        self.img_h, self.img_w = img.shape[0:2]
        if PREPROCESS_TYPE == RESIZE_TYPE:
            # 利用resize的方式进行前处理, 准备nv12的输入数据
            begin_time = time()
            input_tensor = cv2.resize(img, (self.input_W, self.input_H), interpolation=cv2.INTER_NEAREST) # 利用resize重新开辟内存节约一次
            input_tensor = self.bgr2nv12(input_tensor)
            self.y_scale = 1.0 * self.input_H / self.img_h
            self.x_scale = 1.0 * self.input_W / self.img_w
            self.y_shift = 0
            self.x_shift = 0
            logger.info("\033[1;31m" + f"pre process(resize) time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        elif PREPROCESS_TYPE == LETTERBOX_TYPE:
            # 利用 letter box 的方式进行前处理, 准备nv12的输入数据
            begin_time = time()
            self.x_scale = min(1.0 * self.input_H / self.img_h, 1.0 * self.input_W / self.img_w)
            self.y_scale = self.x_scale
            
            if self.x_scale <= 0 or self.y_scale <= 0:
                raise ValueError("Invalid scale factor.")
            
            new_w = int(self.img_w * self.x_scale)
            self.x_shift = (self.input_W - new_w) // 2
            x_other = self.input_W - new_w - self.x_shift
            
            new_h = int(self.img_h * self.y_scale)
            self.y_shift = (self.input_H - new_h) // 2
            y_other = self.input_H - new_h - self.y_shift
            
            input_tensor = cv2.resize(img, (new_w, new_h))
            input_tensor = cv2.copyMakeBorder(input_tensor, self.y_shift, y_other, self.x_shift, x_other, cv2.BORDER_CONSTANT, value=[127, 127, 127])
            input_tensor = self.bgr2nv12(input_tensor)
            logger.info("\033[1;31m" + f"pre process(letter box) time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        else:
            logger.error(f"illegal PREPROCESS_TYPE = {PREPROCESS_TYPE}")
            exit(-1)

        logger.debug("\033[1;31m" + f"pre process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        logger.info(f"y_scale = {self.y_scale:.2f}, x_scale = {self.x_scale:.2f}")
        logger.info(f"y_shift = {self.y_shift:.2f}, x_shift = {self.x_shift:.2f}")
        return input_tensor

    def bgr2nv12(self, bgr_img):
        begin_time = time()
        height, width = bgr_img.shape[0], bgr_img.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        logger.debug("\033[1;31m" + f"bgr8 to nv12 time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return nv12

    def forward(self, input_tensor):
        begin_time = time()
        quantize_outputs = self.quantize_model[0].forward(input_tensor)
        logger.debug("\033[1;31m" + f"forward time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return quantize_outputs

    def c2numpy(self, outputs):
        begin_time = time()
        outputs = [dnnTensor.buffer for dnnTensor in outputs]
        logger.debug("\033[1;31m" + f"c to numpy time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return outputs

    def postProcess(self, outputs):
        begin_time = time()
        # reshape
        s_clses = outputs[0].reshape(-1, self.CLASSES_NUM)
        s_bboxes = outputs[1].reshape(-1, self.REG * 4)
        s_mces = outputs[2].reshape(-1, self.MCES_NUM)

        m_clses = outputs[3].reshape(-1, self.CLASSES_NUM)
        m_bboxes = outputs[4].reshape(-1, self.REG * 4)
        m_mces = outputs[5].reshape(-1, self.MCES_NUM)

        l_clses = outputs[6].reshape(-1, self.CLASSES_NUM)
        l_bboxes = outputs[7].reshape(-1, self.REG * 4)
        l_mces = outputs[8].reshape(-1, self.MCES_NUM)

        protos = outputs[9]


        # classify: 利用numpy向量化操作完成阈值筛选(优化版 2.0)
        s_max_scores = np.max(s_clses, axis=1)
        s_valid_indices = np.flatnonzero(s_max_scores >= self.CONF_THRES_RAW)  # 得到大于阈值分数的索引，此时为小数字
        s_ids = np.argmax(s_clses[s_valid_indices, : ], axis=1)
        s_scores = s_max_scores[s_valid_indices]

        m_max_scores = np.max(m_clses, axis=1)
        m_valid_indices = np.flatnonzero(m_max_scores >= self.CONF_THRES_RAW)  # 得到大于阈值分数的索引，此时为小数字
        m_ids = np.argmax(m_clses[m_valid_indices, : ], axis=1)
        m_scores = m_max_scores[m_valid_indices]

        l_max_scores = np.max(l_clses, axis=1)
        l_valid_indices = np.flatnonzero(l_max_scores >= self.CONF_THRES_RAW)  # 得到大于阈值分数的索引，此时为小数字
        l_ids = np.argmax(l_clses[l_valid_indices, : ], axis=1)
        l_scores = l_max_scores[l_valid_indices]

        # 3个Classify分类分支：Sigmoid计算
        s_scores = 1 / (1 + np.exp(-s_scores))
        m_scores = 1 / (1 + np.exp(-m_scores))
        l_scores = 1 / (1 + np.exp(-l_scores))

        # 3个Bounding Box分支：反量化
        s_bboxes_float32 = s_bboxes[s_valid_indices,:].astype(np.float32) * self.s_bboxes_scale
        m_bboxes_float32 = m_bboxes[m_valid_indices,:].astype(np.float32) * self.m_bboxes_scale
        l_bboxes_float32 = l_bboxes[l_valid_indices,:].astype(np.float32) * self.l_bboxes_scale

        # 3个Bounding Box分支：dist2bbox (ltrb2xyxy)
        s_ltrb_indices = np.sum(softmax(s_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        s_anchor_indices = self.s_anchor[s_valid_indices, :]
        s_x1y1 = s_anchor_indices - s_ltrb_indices[:, 0:2]
        s_x2y2 = s_anchor_indices + s_ltrb_indices[:, 2:4]
        s_dbboxes = np.hstack([s_x1y1, s_x2y2])*8

        m_ltrb_indices = np.sum(softmax(m_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        m_anchor_indices = self.m_anchor[m_valid_indices, :]
        m_x1y1 = m_anchor_indices - m_ltrb_indices[:, 0:2]
        m_x2y2 = m_anchor_indices + m_ltrb_indices[:, 2:4]
        m_dbboxes = np.hstack([m_x1y1, m_x2y2])*16

        l_ltrb_indices = np.sum(softmax(l_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        l_anchor_indices = self.l_anchor[l_valid_indices,:]
        l_x1y1 = l_anchor_indices - l_ltrb_indices[:, 0:2]
        l_x2y2 = l_anchor_indices + l_ltrb_indices[:, 2:4]
        l_dbboxes = np.hstack([l_x1y1, l_x2y2])*32

        # 三个Mask Coefficients分支的反量化
        s_mces_float32 = (s_mces[s_valid_indices,:].astype(np.float32) * self.s_mces_scale)
        m_mces_float32 = (m_mces[m_valid_indices,:].astype(np.float32) * self.m_mces_scale)
        l_mces_float32 = (l_mces[l_valid_indices,:].astype(np.float32) * self.l_mces_scale)

        # Mask Proto的反量化
        protos_float32 = protos.astype(np.float32)[0] * self.mask_scale

        # 大中小特征层阈值筛选结果拼接
        dbboxes = np.concatenate((s_dbboxes, m_dbboxes, l_dbboxes), axis=0)
        scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
        ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)
        mces = np.concatenate((s_mces_float32, m_mces_float32, l_mces_float32), axis=0)

        # xyxy 2 xyhw
        xy = (dbboxes[:,2:4] + dbboxes[:,0:2])/2.0
        hw = (dbboxes[:,2:4] - dbboxes[:,0:2])
        xyhw = np.hstack([xy, hw])

        # 分类别nms
        results = []
        for i in range(self.CLASSES_NUM):
            id_indices = ids==i
            indices = cv2.dnn.NMSBoxes(xyhw[id_indices,:], scores[id_indices], self.SCORE_THRESHOLD, self.NMS_THRESHOLD)
            if len(indices) == 0:
                continue
            for indic in indices:
                x1, y1, x2, y2 = dbboxes[id_indices,:][indic]
                # mask
                x1_corp = int(x1 * self.x_scale_corp)
                y1_corp = int(y1 * self.y_scale_corp)
                x2_corp = int(x2 * self.x_scale_corp)
                y2_corp = int(y2 * self.y_scale_corp)
                # bbox
                x1 = int((x1 - self.x_shift) / self.x_scale)
                y1 = int((y1 - self.y_shift) / self.y_scale)
                x2 = int((x2 - self.x_shift) / self.x_scale)
                y2 = int((y2 - self.y_shift) / self.y_scale)    
                # clip
                x1 = x1 if x1 > 0 else 0
                x2 = x2 if x2 > 0 else 0
                y1 = y1 if y1 > 0 else 0
                y2 = y2 if y2 > 0 else 0
                x1 = x1 if x1 < self.img_w else self.img_w
                x2 = x2 if x2 < self.img_w else self.img_w
                y1 = y1 if y1 < self.img_h else self.img_h
                y2 = y2 if y2 < self.img_h else self.img_h       
                # mask
                mc = mces[id_indices][indic]
                mask = (np.sum(mc[np.newaxis, np.newaxis, :]*protos_float32[y1_corp:y2_corp,x1_corp:x2_corp,:], axis=2) > 0.5).astype(np.uint8)
                # append
                results.append((i, scores[id_indices][indic], x1, y1, x2, y2, mask))

        logger.debug("\033[1;31m" + f"Post Process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")

        return results

coco_names1 = ["background","person","knife","fork","cup","giraffe","plate","table","cake","fence","hat","terrain","tree","car"]
coco_names= ["background","terrain","tree","car-tou","car-wei"]

rdk_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),(49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),(147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]

def draw_detection(img, bbox, score, class_id) -> None:
    """
    Draws a detection bounding box and label on the image.

    Parameters:
        img (np.array): The input image.
        bbox (tuple[int, int, int, int]): A tuple containing the bounding box coordinates (x1, y1, x2, y2).
        score (float): The detection score of the object.
        class_id (int): The class ID of the detected object.
    """
    x1, y1, x2, y2 = bbox
    color = rdk_colors[class_id%20]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{coco_names[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(
        img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
    )
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

if __name__ == "__main__":
    main_map()