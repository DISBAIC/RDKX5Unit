import cv2
import numpy as np

block_color= (0x38, 0x38, 0XFF)
additional_color1=(31, 112, 255)
additional_color2=(29, 178, 255)

def process(image):
    # 检查输入图像
    if image is None:
        print("输入图像为空")
        return None
    
    # 获取图像尺寸
    height, width = image.shape[:2]
    
    # 定义颜色容差
    tolerance = 10
    
    # 创建结果图像，初始为原图的副本
    result_image = image.copy()
    
    # 定义需要处理的颜色列表 (RGB格式) 和对应的缩放因子
    color_processing_params = [
        (block_color, 0.90),          # 原始颜色使用 0.90 缩放因子
        (additional_color1, 0.95),    # 新增颜色1使用 0.91 缩放因子
        (additional_color2, 0.95)     # 新增颜色2使用 0.91 缩放因子
    ]
    
    # 对每种颜色执行处理
    for color, scale_factor in color_processing_params:
        # 将 RGB 转换为 BGR (注意 OpenCV 使用 BGR)
        color_bgr = (color[0], color[1], color[2])
        
        # 创建颜色范围
        lower_bound = np.array([max(0, c - tolerance) for c in color_bgr])
        upper_bound = np.array([min(255, c + tolerance) for c in color_bgr])
        
        # 创建掩码，提取当前颜色
        mask = cv2.inRange(image, lower_bound, upper_bound)
        
        # 从结果图像中删除当前颜色的原始区域
        result_image = cv2.bitwise_and(result_image, result_image, mask=cv2.bitwise_not(mask))
        
        # 使用掩码提取当前颜色图层
        color_layer = cv2.bitwise_and(image, image, mask=mask)
        
        # 计算缩放后的尺寸
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # 缩放颜色图层和掩码
        scaled_color_layer = cv2.resize(color_layer, (new_width, new_height))
        scaled_mask = cv2.resize(mask, (new_width, new_height))
        
        # 计算居中位置
        x_offset = (width - new_width) // 2
        y_offset = (height - new_height) // 2
        
        # 将缩放后的颜色图层添加到结果图像的中心区域
        roi = result_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width]
        scaled_color_only = cv2.bitwise_and(scaled_color_layer, scaled_color_layer, mask=scaled_mask)
        roi_without_scaled_color = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(scaled_mask))
        
        # 合并图像
        result_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = cv2.add(roi_without_scaled_color, scaled_color_only)
    
    return result_image




# 如果直接运行此文件，执行处理
if __name__ == "__main__":
    # 读取图像
    input_image = cv2.imread('test.jpg')
    if input_image is None:
        print("无法读取 test.jpg 文件")
    else:
        # 处理图像
        result = process(input_image)
        
        if result is not None:
            # 显示结果
            cv2.imshow('Original Image', input_image)
            cv2.imshow('Processed Result', result)
            
            # 保存结果
            cv2.imwrite('processed_result.jpg', result)
            
            print("处理完成！")
            print(f"原图尺寸: {input_image.shape[1]}x{input_image.shape[0]}")
            print("已保存: processed_result.jpg")
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()

