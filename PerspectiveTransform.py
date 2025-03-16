# -*- coding:utf-8 -*-
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
def cal_perspective_params(img, points):
    height, width = img.shape[:2]

    # src 和 dst 對應四個點，用來計算透視變換矩陣
    src = np.float32(points)
    width_top = np.linalg.norm(points[1] - points[0])  
    width_bottom = np.linalg.norm(points[2] - points[3])  
    width = max(int(width_top), int(width_bottom))  
    height_left = np.linalg.norm(points[3] - points[0])  
    height_right = np.linalg.norm(points[2] - points[1])  
    height = max(int(height_left), int(height_right))  

    dst = np.float32([[0,0], [width,0], [width,height], [0,height]])
    # 透视矩阵
    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(img, M, (width, height))
    
    return M, warped_img
# 计算透视变换参数矩阵
input_folder = './imprint/al/'
output_folder = './imprint/al/transform/'
# 確保輸出資料夾存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# 取得所有 PNG 圖片檔案
image_paths = sorted(glob.glob(os.path.join(input_folder, '*.png')))
# 定義透視變換的四個點
points = np.array([(136, 0), (508, 0), (457, 345), (203, 348)])
for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        print(f"讀取失敗: {img_path}")
        continue
    # 計算透視變換
    M, transform_img = cal_perspective_params(img, points)
    # 顯示原始圖像與變換後圖像
    cv2.imshow('Original Image', img)
    cv2.imshow('Transformed Image', transform_img)
    # 儲存結果
    output_path = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(output_path, transform_img)
    # 顯示 1 秒後切換到下一張
    cv2.waitKey(1000)  
cv2.destroyAllWindows()
######################################
