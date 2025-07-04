import os
import numpy as np
import cv2
import config

# 添加 OpenCV 類型提示以解決 linter 錯誤
from typing import Any
cv2: Any

def detect_circles(diff_img, color_img, output_path_prefix):
    """
    對相減後的圖片進行圓形檢測
    :param warped_diff_img: 經過透視變換的差異圖片
    :param warped_color_img: 經過透視變換的彩色原圖
    :param output_path_prefix: 輸出圖片的路徑前綴
    :return: 圓心座標列表和處理後的圖片
    """
    gray_blurred = cv2.GaussianBlur(diff_img, (5,5), 2)
    cv2.imwrite(f"{output_path_prefix}_gray_blurred.png", gray_blurred)

    # 使用 Canny 邊緣檢測
    canny = cv2.Canny(gray_blurred, 40,80)
    cv2.imwrite(f"{output_path_prefix}_canny.png", canny)
    
    # 使用篩選後的邊緣進行霍夫圓變換
    circles = cv2.HoughCircles(canny, 
                               cv2.HOUGH_GRADIENT, 
                               dp=1.2, 
                               minDist=100,
                               param1=20, 
                               param2=30, 
                               minRadius=10, 
                               maxRadius=30)
   
    # 使用彩色原圖作為輸出圖片
    output_img = color_img.copy()
    centers = []
    if circles is not None:
        # 把 shape (1,1,3) ➜ (1,3)，然後轉 int
        x, y, r = np.round(circles[0, 0]).astype(int)
        centers.append((x, y))
        print(f"圓心 = ({x}, {y})，半徑 = {r}")
        # 畫在原圖上
        cv2.circle(output_img, (x, y), r, (0, 255, 0), 2)   # 外框
        cv2.circle(output_img, (x, y), 2, (0, 0, 255), 3)   # 圓心
    else:
        print("沒找到圓形")
    cv2.imwrite(f"{output_path_prefix}_detected.png", output_img)
    return centers, output_img


if __name__ == "__main__":
    # 檢查並創建輸出目錄
    output_folder = "./estimation_1/circle"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已創建目錄：{output_folder}")
    
    # 基準圖片
    base_img = cv2.imread("./estimation_1/img9.png")
    if base_img is None:
        print("無法讀取基準圖片")
        exit()
  
    # 要處理的樣本圖片列表
    sample_images = [
        "./estimation_1/img0.png",
        "./estimation_1/img1.png",
        "./estimation_1/img2.png",
        "./estimation_1/img3.png",
        "./estimation_1/img4.png",
        "./estimation_1/img5.png",
        "./estimation_1/img6.png",
        "./estimation_1/img7.png",
        "./estimation_1/img8.png",
    ]
    
    # 儲存所有檢測到的圓心座標
    all_circle_centers = []
    
    # 轉換基準圖片為灰度圖
    base_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    
    # 處理每張樣本圖片
    for sample_path in sample_images:
        print(f"\n處理圖片：{sample_path}")
        sample_img = cv2.imread(sample_path)
        if sample_img is None:
            print(f"無法讀取圖片：{sample_path}")
            continue
        
        # 轉換為灰度圖並相減
        sample_gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
        diff_img = cv2.absdiff(base_gray, sample_gray)
        
        # 檢測圓形
        basename = os.path.splitext(os.path.basename(sample_path))[0]
        output_path_prefix = os.path.join(output_folder, basename)
        circle_centers, result_img = detect_circles(diff_img, sample_img, output_path_prefix)
        
        if circle_centers:
            # 將檢測到的圓心座標添加到總列表中
            all_circle_centers.extend(circle_centers)
            print(f"在 {sample_path} 中檢測到 {len(circle_centers)} 個圓形")
    
    def order_four(pts):
        """
        輸入 4 個 (x, y)；輸出依 左上→右上→左下→右下 排序後的 4 個 (x, y)。
        """
        # 先依 y 再依 x 排出「上 2、下 2」
        pts = sorted(pts, key=lambda p: (p[1], p[0]))
        top_two    = sorted(pts[:2], key=lambda p: p[0])  # 左上、右上
        bottom_two = sorted(pts[2:], key=lambda p: p[0])  # 左下、右下
        return top_two + bottom_two
    
    ordered_centers = []
    for i in range(0, len(all_circle_centers), 4):
        ordered_centers.extend(order_four(all_circle_centers[i:i+4]))

    all_circle_centers = ordered_centers 
        
    #####################################################################################
    ############################# 寫回 config.py ########################################
    #####################################################################################
    if all_circle_centers:
        
        circles_list = [[round(float(x), 3), round(float(y), 3)] for x, y in all_circle_centers]
        with open('config.py', 'r', encoding='utf-8') as file:
            lines = file.readlines()
        # 檢查是否已存在 CALIB_CIRCLES_PTS
        for i, line in enumerate(lines): 
            if line.startswith('CALIB_CIRCLES_PTS'):
                # 使用格式化字符串確保三位小數
                lines[i] = f'CALIB_CIRCLES_PTS = {circles_list}\n'
                break
        else:
            lines.append(f'CALIB_CIRCLES_PTS = {circles_list}\n')
        # 寫回文件
        with open('config.py', 'w', encoding='utf-8') as file:
            file.writelines(lines)
        
        print("\n已更新 config.py 中的 CALIB_CIRCLES_PTS")
        print(f"總共檢測到 {len(all_circle_centers)} 個圓形中心點")
        print("座標列表（三位小數）:")
        for i, (x, y) in enumerate(circles_list):
            print(f"圓心 #{i}: [{x:.3f}, {y:.3f}]")
    else:
        print("\n未檢測到任何圓形")

