# -*- coding:utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 计算透视变换参数矩阵
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

    dst = np.float32([[0,0],[width,0],[width,height],[0,height]])
    # 透视矩阵
    M = cv2.getPerspectiveTransform(src, dst)
    print(M)
    warped_img = cv2.warpPerspective(img, M, (width, height))
    return M, warped_img

if __name__ == '__main__':
    img = cv2.imread('./imprint/al/img4.png')
    points = np.array([(130,0),(520,0),(470,360),(185,360)])
    M, transform_img = cal_perspective_params(img, points)
    cv2.imshow('img',img)
    cv2.imshow('trasform_img0',transform_img)
    cv2.imwrite('trasform_img4.png',transform_img)
    cv2.waitKey(0)
