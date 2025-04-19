# -*- coding:utf-8 -*-
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np

def ROI(img, points):
    pts = np.array([points])
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.polylines(mask, pts, 1, 255)    
    cv2.fillPoly(mask, pts, 255)    
    dst = cv2.bitwise_and(img, img, mask=mask)
    bg = np.ones_like(img, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)  
    
    # 計算 ROI 的邊界框
    x, y, w, h = cv2.boundingRect(pts)
    cropped_roi = dst[y:y+h, x:x+w]
    # 建立白色背景並應用 mask
    bg = np.ones_like(img, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst_white = bg + dst
    # 裁剪白色背景的 ROI
    cropped_dst_white = dst_white[y:y+h, x:x+w]
    cv2.imshow("cropped_dst_white", cropped_dst_white)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cropped_dst_white

def detect_points(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    cv2.imshow("edges", edges)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)
    cv2.imshow("contour_image", contour_image)
     # 變數用來存儲最大面積矩形
    largest_rectangle = None
    max_area = 0
    h, w = image.shape[:2]
    border_threshold = 10
    # 重新篩選矩形，排除接觸邊界的輪廓
    for contour in contours:
        # 逼近多邊形
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # 只保留四邊形
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            x, y, w_rect, h_rect = cv2.boundingRect(approx)  # 取得矩形邊界框
            # 過濾掉接觸圖片邊界的矩形
            if (x < border_threshold or y < border_threshold or
                x + w_rect > w - border_threshold or y + h_rect > h - border_threshold):
                continue  # 跳過這個輪廓
            if area > max_area :
                max_area = area
                largest_rectangle = approx
    # 在圖像上繪製最大矩形
    final_image = image.copy()
    if largest_rectangle is not None:
        cv2.drawContours(final_image, [largest_rectangle], -1, (0, 255, 0), 3)
    cv2.imshow("final_image", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if largest_rectangle is not None:
        largest_rect_points = largest_rectangle.reshape(-1, 2)
         # 依照 (x+y) 值排序以確保點順序正確
        # (左上, 右上, 左下, 右下)
        sorted_points = sorted(largest_rect_points, key=lambda p: (p[1], p[0]))  # 先依 y 排序，再依 x 排序

        # 確保順序為: 左上、右上、左下、右下
        if sorted_points[0][0] > sorted_points[1][0]:
            sorted_points[0], sorted_points[1] = sorted_points[1], sorted_points[0]  # 確保左上在前
        if sorted_points[2][0] < sorted_points[3][0]:
            sorted_points[2], sorted_points[3] = sorted_points[3], sorted_points[2]  # 確保右下在前
            
        print("最大矩形的四個角點座標:")
        for i, point in enumerate(sorted_points):
            print(f"Point {i+1}: {point}")
        return sorted_points
    else:
        print("未找到矩形")
    

    # # **計算邊長**
    # def euclidean_distance(pt1, pt2):
    #     return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    # top_edge = euclidean_distance(sorted_points[0], sorted_points[1])  # 左上 → 右上
    # right_edge = euclidean_distance(sorted_points[1], sorted_points[2])  # 右上 → 右下
    # bottom_edge = euclidean_distance(sorted_points[2], sorted_points[3])  # 右下 → 左下
    # left_edge = euclidean_distance(sorted_points[3], sorted_points[0])  # 左下 → 左上

    # print("四條邊的長度（像素）:")
    # print(f"上邊（左上 → 右上）: {top_edge:.2f} pixels")
    # print(f"右邊（右上 → 右下）: {right_edge:.2f} pixels")
    # print(f"下邊（右下 → 左下）: {bottom_edge:.2f} pixels")
    # print(f"左邊（左下 → 左上）: {left_edge:.2f} pixels")

def apply_perspective_transform(image, sorted_points):
    """
    對影像應用透視變換，將偵測到的四邊形轉換為標準矩形
    :param image: 原始影像
    :param sorted_points: 四個角點座標 (左上、右上、左下、右下)
    :return: 透視變換後的影像
    """
    # 轉換為 NumPy 陣列
    sorted_points = np.array(sorted_points, dtype=np.float32)

    # 計算透視變換後的寬度和高度
    width_top = np.linalg.norm(sorted_points[1] - sorted_points[0])  # 右上 - 左上
    width_bottom = np.linalg.norm(sorted_points[2] - sorted_points[3])  # 右下 - 左下
    width = int((width_top + width_bottom)//2)

    height_left = np.linalg.norm(sorted_points[3] - sorted_points[0])  # 左下 - 左上
    height_right = np.linalg.norm(sorted_points[2] - sorted_points[1])  # 右下 - 右上
    height = int((height_left + height_right)//2)
    
    # 定義變換後的四個點 (標準矩形)
    dst_points = np.array([
        [0, 0],          # 左上角
        [200-1, 0],    # 右上角
        [200-1, 250-1],  # 右下角
        [0, 250-1],   # 左下角
    ], dtype=np.float32)

    # 計算透視變換矩陣
    H = cv2.getPerspectiveTransform(sorted_points, dst_points)

    # 應用透視變換
    warped_image = cv2.warpPerspective(image, H, (200, 250))
    cv2.imshow("warped_image",warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return warped_image, H

def is_square(points, tolerance=5):
    """
    根據四個點判斷是否為正方形
    :param points: 4 個點的 NumPy 陣列 (左上、右上、左下、右下)
    :param tolerance: 容許的誤差範圍 (像素)，避免因數值誤差導致錯誤判斷
    :return: True (是正方形) 或 False (不是正方形)
    """
    if points is None or len(points) != 4:
        return False

    # 取四個點
    p1, p2, p3, p4 = points  # 左上、右上、右下、左下

    # 計算四條邊長
    side1 = np.linalg.norm(p2 - p1)  # 上邊
    side2 = np.linalg.norm(p3 - p2)  # 右邊
    side3 = np.linalg.norm(p3 - p4)  # 下邊
    side4 = np.linalg.norm(p4 - p1)  # 左邊
    print(side1,side2,side3,side4)
    if (abs(side1-side3) < tolerance or 
        abs(side2-side4) < tolerance or
        abs(side1-side2) < tolerance or
        abs(side3-side4) < tolerance):
        return True
    else: return False

######################################
# 计算透视变换参数矩阵
img_path= './calibration/perspective/img1_trans.png'

# 定義透視變換的四個點
points = np.array([(120, 0), (506, 0), (458, 366), (197, 369)]) # 框偵測的四個點
img = cv2.imread(img_path)
cropped_img = ROI(img, points)
sorted_points = detect_points(cropped_img)
transformed_img, H = apply_perspective_transform(cropped_img, sorted_points)
inner_square_pts = detect_points(transformed_img)

if is_square(inner_square_pts):
    np.save("./calibration/perspective_matrix.npy", H)
    print("save perspective_matrix.npy")
else:
    print("not square")
