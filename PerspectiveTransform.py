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
img_path = "F:/img0_transform.png"
# 定義透視變換的四個點
points = np.array([(136, 0), (508, 0), (457, 345), (203, 348)])
img = cv2.imread(img_path)
 # 計算透視變換
M, transform_img = cal_perspective_params(img, points)
 # 顯示原始圖像與變換後圖像
cv2.imshow('Original Image', img)
cv2.imshow('Transformed Image', transform_img)
cv2.waitKey(0)  
cv2.destroyAllWindows()
######################################
