import cv2
import numpy as np
import os
from pathlib import Path

def create_mask(base_img, sample_img, threshold=30, noise_size=1):
    """
    创建二值掩码
    """
    # 计算差异
    diff = cv2.absdiff(base_img, sample_img)
    
    # 转换为灰度图
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    _, binary_diff = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    
    # 噪声过滤
    kernel_size = 2 * noise_size + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    filtered_mask = cv2.morphologyEx(binary_diff, cv2.MORPH_OPEN, kernel)
    
    return filtered_mask

def process_images(input_dir, output_dir, threshold=30, noise_size=1):
    """
    处理图像并创建数据集
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    
    # 获取所有图像文件
    base_path = os.path.join(input_dir, 'img0_base_RGB.png')
    if not os.path.exists(base_path):
        print(f"错误：找不到基准图像 {base_path}")
        return
    
    base_img = cv2.imread(base_path)
    
    # 处理所有样本图像
    for img_path in Path(input_dir).glob('img*_RGB.png'):
        if 'base' in str(img_path):
            continue
            
        # 读取样本图像
        sample_img = cv2.imread(str(img_path))
        if sample_img is None:
            print(f"警告：无法读取图像 {img_path}")
            continue
            
        # 创建掩码
        mask = create_mask(base_img, sample_img, threshold, noise_size)
        
        # 保存结果
        img_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_dir, 'images', img_name), sample_img)
        cv2.imwrite(os.path.join(output_dir, 'masks', img_name), mask)
        
        print(f"已处理: {img_name}")

def main():
    # 设置输入输出路径
    input_dir = "C:/Jill/Code/camera/imprint/al_grip/RGB/transform"
    output_dir = "C:/Jill/Code/camera/imprint/al_grip/RGB/dataset"
    
    # 处理参数
    threshold = 30
    noise_size = 1
    
    # 处理图像
    process_images(input_dir, output_dir, threshold, noise_size)
    print("数据集创建完成！")

if __name__ == "__main__":
    main() 