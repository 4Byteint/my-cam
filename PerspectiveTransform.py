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
points = np.array([(127, 15), (495, 8), (438, 346), (189, 348)])
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
# import cv2
# import numpy as np

# # 讀取圖像
# image = cv2.imread('./calibration/img0_transform.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # 進行二值化處理
# _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

# # 找輪廓
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(gray, contours, -1, (0, 255, 0), 2)
# # 顯示結果
# cv2.imshow('Contours', gray)
# # 遍歷輪廓，找出接近梯形的形狀
# for cnt in contours:
#     # 近似多邊形
#     epsilon = 0.02 * cv2.arcLength(cnt, True)
#     approx = cv2.approxPolyDP(cnt, epsilon, True)

#     # 如果找到四個點，則可能是梯形
#     if len(approx) == 4:
#         # 畫出梯形
#         cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)

#         # 印出四個點的座標
#         for i, point in enumerate(approx):
#             x, y = point[0]
#             print(f"Point {i + 1}: ({x}, {y})")
#             cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

# # 顯示結果
# cv2.imshow('Detected Rectangles', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
