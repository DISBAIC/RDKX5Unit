import sys
import signal
import cv2
import numpy as np
from scipy.special import softmax
from scipy.special import expit as sigmoid
from time import time
import bpu_infer_lib  # Model Zoo Python API
from typing import Tuple
import os
import ctypes
from hobot_vio import libsrcampy as srcampy
import json

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

# def bgr2nv12_opencv(image):
#     height, width = image.shape[0], image.shape[1]
#     area = height * width
#     yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
#     y = yuv420p[:area]
#     uv_planar = yuv420p[area:].reshape((2, area // 4))
#     uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

#     nv12 = np.zeros_like(yuv420p)
#     nv12[:height * width] = y
#     nv12[height * width:] = uv_packed
#     return nv12

class BPU_Detect:
    def __init__(self, model_path:str,
                labelnames:list,
                num_classes:int = None,
                conf:float = 0.45,
                iou:float = 0.45,
                anchors:np.array = np.array([
                    [10,13, 16,30, 33,23],  # P3/8
                    [30,61, 62,45, 59,119],  # P4/16
                    [116,90, 156,198, 373,326],  # P5/32
                   ]),
                strides = np.array([8, 16, 32]),
                mode:bool = False,
                is_save:bool = True
                ):
        self.model = model_path
        self.labelname = labelnames
        self.inf = bpu_infer_lib.Infer(False)
        self.inf.load_model(self.model)
        self.conf = conf
        self.iou = iou
        self.anchors = anchors
        self.strides = strides
        self.input_w = 640
        self.input_h = 640
        self.nc = num_classes if num_classes is not None else len(self.labelname)
        self.mode = mode
        self.is_save = is_save
        self._init_grids()
        
    def _init_grids(self) :
        """初始化特征图网格"""
        def _create_grid(stride: int) :
            """创建单个stride的网格和anchors"""
            grid = np.stack([
                np.tile(np.linspace(0.5, self.input_w//stride - 0.5, self.input_w//stride), 
                       reps=self.input_h//stride),
                np.repeat(np.arange(0.5, self.input_h//stride + 0.5, 1), 
                         self.input_w//stride)
            ], axis=0).transpose(1,0)
            grid = np.hstack([grid] * 3).reshape(-1, 2)
            
            anchors = np.tile(
                self.anchors[int(np.log2(stride/8))], 
                self.input_w//stride * self.input_h//stride
            ).reshape(-1, 2)
            
            return grid, anchors
            
        # 创建不同尺度的网格
        self.s_grid, self.s_anchors = _create_grid(self.strides[0])
        self.m_grid, self.m_anchors = _create_grid(self.strides[1]) 
        self.l_grid, self.l_anchors = _create_grid(self.strides[2])
        
        # print(f"网格尺寸: {self.s_grid.shape = }  {self.m_grid.shape = }  {self.l_grid.shape = }")
        # print(f"Anchors尺寸: {self.s_anchors.shape = }  {self.m_anchors.shape = }  {self.l_anchors.shape = }")



    def bgr2nv12_opencv(self, image):
        height, width = image.shape[0], image.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        return  nv12
    
    # def PreProcess(self, img, method=0):
    #     """
    #     预处理函数
    #     Args:
    #         img: 输入图像或图片路径
    #         method: 选择使用的方法
    #             - 0: 使用read_input
    #             - 1: 使用read_img_to_nv12
    #     """
    #     # 获取原始图片和尺寸
    #     if isinstance(img, str):
    #         # 输入是图片路径
    #         if method == 1:
    #             # 先获取原始图片尺寸
    #             orig_img = cv2.imread(img)
    #             if orig_img is None:
    #                 raise ValueError(f"无法读取图片: {img}")
    #             img_h, img_w = orig_img.shape[0:2]
    #             self.inf.read_img_to_nv12(img, 0)# 使用API的方法直接读取
    #         elif method == 0:
    #             # method == 0，读取图片后处理
    #             orig_img = cv2.imread(img)
    #             if orig_img is None:
    #                 raise ValueError(f"无法读取图片: {img}")
    #             input_tensor = cv2.resize(orig_img, (self.input_h, self.input_w))
    #             input_tensor = self.bgr2nv12_opencv(input_tensor)
    #             self.inf.read_input(input_tensor, 0)
    #             img_h, img_w = orig_img.shape[0:2]
    #     else:
    #         print("输入格式有误")
    #         return False
        
    #     # 计算缩放比例
    #     self.y_scale = img_h / self.input_h
    #     self.x_scale = img_w / self.input_w
        
    #     print(f"原始尺寸: {img_w}x{img_h}, 输入尺寸: {self.input_w}x{self.input_h}")
    #     print(f"缩放比例: x_scale={self.x_scale}, y_scale={self.y_scale}")
        
    #     return True

    def PreProcess(self, img, method=0):
    # """
    # 预处理函数
    # Args:
    #     img: 输入图像或图片路径
    #     method: 选择使用的方法
    #         - 0: 使用read_input（默认）
    #         - 1: 使用read_img_to_nv12（仅支持路径）
    # """
        if isinstance(img, str):
            # 如果是图片路径，仍然可以保留方法1
            if method == 1:
                orig_img = cv2.imread(img)
                if orig_img is None:
                    raise ValueError(f"无法读取图片: {img}")
                img_h, img_w = orig_img.shape[0:2]
                self.inf.read_img_to_nv12(img, 0)
            elif method == 0:
                orig_img = cv2.imread(img)
                if orig_img is None:
                    raise ValueError(f"无法读取图片: {img}")
                input_tensor = cv2.resize(orig_img, (self.input_h, self.input_w))
                input_tensor = self.bgr2nv12_opencv(input_tensor)
                self.inf.read_input(input_tensor, 0)
                img_h, img_w = orig_img.shape[0:2]
        else:
            # 直接传入图像数组
            orig_img = img.copy()
            input_tensor = cv2.resize(orig_img, (self.input_h, self.input_w))
            input_tensor = self.bgr2nv12_opencv(input_tensor)
            self.inf.read_input(input_tensor, 0)
            img_h, img_w = orig_img.shape[0:2]

        # 计算缩放比例
        self.y_scale = img_h / self.input_h
        self.x_scale = img_w / self.input_w
        # 计算缩放比例
        # print(f"原始尺寸: {img_w}x{img_h}, 输入尺寸: {self.input_w}x{self.input_h}")
        # print(f"缩放比例: x_scale={self.x_scale}, y_scale={self.y_scale}")

        return True

    def PostPrecess(self, method=1):
        """
        后处理函数
        Args:
            method: 选择使用的方法
                - '0': 使用get_output
                - '1': 使用get_infer_res_np_float32
        """
        if method == 1 :
            # 方法1：使用get_infer_res_np_float32获取原始输出并处理
            # print("\n=== 方法1: 使用get_infer_res_np_float32 ===")
            s_pred = self.inf.get_infer_res_np_float32(0)
            m_pred = self.inf.get_infer_res_np_float32(1)
            l_pred = self.inf.get_infer_res_np_float32(2)
            # print(f"原始输出: {s_pred.shape = }  {m_pred.shape = }  {l_pred.shape = }")

            # reshape
            s_pred = s_pred.reshape([-1, (5 + self.nc)])
            m_pred = m_pred.reshape([-1, (5 + self.nc)])
            l_pred = l_pred.reshape([-1, (5 + self.nc)])
            # print(f"Reshape后: {s_pred.shape = }  {m_pred.shape = }  {l_pred.shape = }")

            # classify: 利用numpy向量化操作完成阈值筛选
            s_raw_max_scores = np.max(s_pred[:, 5:], axis=1)
            s_max_scores = 1 / ((1 + np.exp(-s_pred[:, 4]))*(1 + np.exp(-s_raw_max_scores)))
            s_valid_indices = np.flatnonzero(s_max_scores >= self.conf)
            s_ids = np.argmax(s_pred[s_valid_indices, 5:], axis=1)
            s_scores = s_max_scores[s_valid_indices]

            m_raw_max_scores = np.max(m_pred[:, 5:], axis=1)
            m_max_scores = 1 / ((1 + np.exp(-m_pred[:, 4]))*(1 + np.exp(-m_raw_max_scores)))
            m_valid_indices = np.flatnonzero(m_max_scores >= self.conf)
            m_ids = np.argmax(m_pred[m_valid_indices, 5:], axis=1)
            m_scores = m_max_scores[m_valid_indices]

            l_raw_max_scores = np.max(l_pred[:, 5:], axis=1)
            l_max_scores = 1 / ((1 + np.exp(-l_pred[:, 4]))*(1 + np.exp(-l_raw_max_scores)))
            l_valid_indices = np.flatnonzero(l_max_scores >= self.conf)
            l_ids = np.argmax(l_pred[l_valid_indices, 5:], axis=1)
            l_scores = l_max_scores[l_valid_indices]

            # 特征解码
            s_dxyhw = 1 / (1 + np.exp(-s_pred[s_valid_indices, :4]))
            s_xy = (s_dxyhw[:, 0:2] * 2.0 + self.s_grid[s_valid_indices,:] - 1.0) * self.strides[0]
            s_wh = (s_dxyhw[:, 2:4] * 2.0) ** 2 * self.s_anchors[s_valid_indices, :]
            s_xyxy = np.concatenate([s_xy - s_wh * 0.5, s_xy + s_wh * 0.5], axis=-1)

            m_dxyhw = 1 / (1 + np.exp(-m_pred[m_valid_indices, :4]))
            m_xy = (m_dxyhw[:, 0:2] * 2.0 + self.m_grid[m_valid_indices,:] - 1.0) * self.strides[1]
            m_wh = (m_dxyhw[:, 2:4] * 2.0) ** 2 * self.m_anchors[m_valid_indices, :]
            m_xyxy = np.concatenate([m_xy - m_wh * 0.5, m_xy + m_wh * 0.5], axis=-1)

            l_dxyhw = 1 / (1 + np.exp(-l_pred[l_valid_indices, :4]))
            l_xy = (l_dxyhw[:, 0:2] * 2.0 + self.l_grid[l_valid_indices,:] - 1.0) * self.strides[2]
            l_wh = (l_dxyhw[:, 2:4] * 2.0) ** 2 * self.l_anchors[l_valid_indices, :]
            l_xyxy = np.concatenate([l_xy - l_wh * 0.5, l_xy + l_wh * 0.5], axis=-1)

            # 大中小特征层阈值筛选结果拼接
            xyxy = np.concatenate((s_xyxy, m_xyxy, l_xyxy), axis=0)
            scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
            ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)

        elif method == 0:
            # 方法2：使用get_output获取输出
            print("\n=== 方法2: 使用get_output ===")
            if not self.inf.get_output():
                raise RuntimeError("获取输出失败")
            
            classes_scores = self.inf.outputs[0].data  # (1, 80, 80, 18)
            bboxes = self.inf.outputs[1].data         # (1, 40, 40, 18)
            print(f"classes_scores: shape={classes_scores.shape}")
            print(f"bboxes: shape={bboxes.shape}")
            
            # 直接使用4D数据
            # 每个网格有3个anchor，每个anchor预测6个值(4个框坐标+1个objectness+1个类别)
            batch, height, width, channels = classes_scores.shape
            num_anchors = 3
            pred_per_anchor = 6
            
            scores_list = []
            boxes_list = []
            ids_list = []
            # 处理每个网格点
            for h in range(height):
                for w in range(width):
                    for a in range(num_anchors):
                        # 获取当前anchor的预测值
                        start_idx = int(a * pred_per_anchor)
                        box = classes_scores[0, h, w, start_idx:start_idx+4].copy()  # 框坐标
                        obj_score = float(classes_scores[0, h, w, start_idx+4])      # objectness
                        cls_score = float(classes_scores[0, h, w, start_idx+5])      # 类别分数
                        # sigmoid激活
                        obj_score = 1 / (1 + np.exp(-obj_score))
                        cls_score = 1 / (1 + np.exp(-cls_score))
                        score = obj_score * cls_score
                        
                        # 如果分数超过阈值，保存这个预测
                        if score >= self.conf:
                            # 解码框坐标
                            box = 1 / (1 + np.exp(-box))  # sigmoid
                            cx = float((box[0] * 2.0 + w - 0.5) * self.strides[0])
                            cy = float((box[1] * 2.0 + h - 0.5) * self.strides[0])
                            w_pred = float((box[2] * 2.0) ** 2 * self.anchors[0][a*2])
                            h_pred = float((box[3] * 2.0) ** 2 * self.anchors[0][a*2+1])
                            
                            # 转换为xyxy格式
                            x1 = cx - w_pred/2
                            y1 = cy - h_pred/2
                            x2 = cx + w_pred/2
                            y2 = cy + h_pred/2
                            
                            boxes_list.append([x1, y1, x2, y2])
                            scores_list.append(float(score))  # 确保是标量
                            ids_list.append(0)  # 假设只有一个类别
            if boxes_list:
                xyxy = np.array(boxes_list, dtype=np.float32)
                scores = np.array(scores_list, dtype=np.float32)
                ids = np.array(ids_list, dtype=np.int32)
            else:
                xyxy = np.array([], dtype=np.float32).reshape(0, 4)
                scores = np.array([], dtype=np.float32)
                ids = np.array([], dtype=np.int32)

        else:
            raise ValueError("method must be 0 or 1")

        # NMS处理
        indices = cv2.dnn.NMSBoxes(xyxy.tolist(), scores.tolist(), self.conf, self.iou)
        
        if len(indices) > 0:
            indices = np.array(indices).flatten()
            self.bboxes = (xyxy[indices] * np.array([self.x_scale, self.y_scale, self.x_scale, self.y_scale])).astype(np.int32)
            self.scores = scores[indices]
            self.ids = ids[indices]
        else:
            print("No detections after NMS")
            self.bboxes = np.array([], dtype=np.int32).reshape(0, 4)
            self.scores = np.array([], dtype=np.float32)
            self.ids = np.array([], dtype=np.int32)



    def draw_detection(self,img: np.array, 
                      box,
                      score: float, 
                      class_id: int,
                      labelname: list):
        x1, y1, x2, y2 = box
        rdk_colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 青色
        ]
        color = rdk_colors[class_id % len(rdk_colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{labelname[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(
            img, 
            (label_x, label_y - label_height), 
            (label_x + label_width, label_y + label_height), 
            color, 
            cv2.FILLED
        )
        cv2.putText(img, label, (label_x, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
    # def detect_result(self, img):
    #     if isinstance(img, str):
    #         draw_img = cv2.imread(img)
    #     else:
    #         draw_img = img.copy()
    #     for class_id, score, bbox in zip(self.ids, self.scores, self.bboxes):
    #         x1, y1, x2, y2 = bbox
    #         print("(%d, %d, %d, %d) -> %s: %.2f"%(x1,y1,x2,y2, self.labelname[class_id], score))
    #         if self.is_save:
    #             self.draw_detection(draw_img, (x1, y1, x2, y2), score, class_id, self.labelname)
    #     if self.is_save:
    #             cv2.imwrite("result.jpg", draw_img)

    def detect_result(self, img):
        if isinstance(img, str):
            draw_img = cv2.imread(img)
        else:
            draw_img = img.copy()

        for class_id, score, bbox in zip(self.ids, self.scores, self.bboxes):
            x1, y1, x2, y2 = bbox
            print("(%d, %d, %d, %d) -> %s: %.2f"%(x1,y1,x2,y2, self.labelname[class_id], score))
            self.draw_detection(draw_img, (x1, y1, x2, y2), score, class_id, self.labelname)

        # 返回绘制好的图像
        cv2.imshow("fire",draw_img)
        return draw_img

    def detect(self, img, method_pre=0, method_post=1):
        """
        检测函数
        Args:
            img_path: 图片路径或图片数组
            method_pre: 预处理方法
                - 0: 使用read_input（默认，读取图片后处理）
                - 1: 使用read_img_to_nv12（直接读取路径）
            method_post: 后处理方法
        """
        # 预处理
        self.PreProcess(img, method=method_pre)
        
        # 推理和后处理
        self.inf.forward(self.mode)
        self.PostPrecess(method_post)
        self.detect_result(img)
        

def main_fire():
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

        coconame = ["fire","smoke"]
        # test_img = "/app/my_project/photo/0ce61215-6d34-49f9-aa6d-5f0feed430f8.jpg"
        models = "/app/my_project/bin/converted_model2.bin"
        # infer = BPU_Detect(models,coconame,conf=0.1,iou=0.3)
        infer = BPU_Detect(models,coconame,conf=0.2,iou=0.3,mode = True)
        # infer.detect(test_img,method_pre=1,method_post=1)

        while True:

            _ ,frame = cap.read()
            
            # print(frame.shape)

            if frame is None:
                print("Failed to get image from usb camera")
                continue
            
            des_dim = (1024, 768)
            resized_data = cv2.resize(frame, des_dim, interpolation=cv2.INTER_AREA)

            print("Image shape:", resized_data.shape)

            t0 = time()
            # Forward
            # outputs = models[0].forward(nv12_data)
            
            # print("forward time is :", (t1 - t0))

        
            infer.detect(resized_data,method_pre=1,method_post=1)

            t1 = time()

            print("forward time is :", (t1 - t0))
            
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC键或q键
                break

if __name__ == "__main__":
    main_fire()

        
  