# -*- coding:utf-8 -*-
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # 计算透视变换参数矩阵
# def cal_perspective_params(img, points):
#     height, width = img.shape[:2]

#     # src 和 dst 對應四個點，用來計算透視變換矩陣
#     src = np.float32(points)
#     width_top = np.linalg.norm(points[1] - points[0])  
#     width_bottom = np.linalg.norm(points[2] - points[3])  
#     width = max(int(width_top), int(width_bottom))  
#     height_left = np.linalg.norm(points[3] - points[0])  
#     height_right = np.linalg.norm(points[2] - points[1])  
#     height = max(int(height_left), int(height_right))  

#     dst = np.float32([[0,0],[width,0],[width,height],[0,height]])
#     # 透视矩阵
#     M = cv2.getPerspectiveTransform(src, dst)
#     print(M)
#     warped_img = cv2.warpPerspective(img, M, (width, height))
#     return M, warped_img

# if __name__ == '__main__':
#     img = cv2.imread('./imprint/al_calib/img4.png')
#     points = np.array([(136,0),(515,0),(458,340),(200,348)])
#     M, transform_img = cal_perspective_params(img, points)
#     cv2.imshow('img',img)
#     cv2.imshow('transform_img0',transform_img)
#     #cv2.imwrite('transform_img4.png',transform_img)
#     cv2.waitKey(0)
######################################
import cv2
import numpy as np

# 讀取圖像
image = cv2.imread('./calibration/img0_transform.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 進行二值化處理
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 找輪廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 遍歷輪廓，找出接近梯形的形狀
for cnt in contours:
    # 近似多邊形
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # 如果找到四個點，則可能是梯形
    if len(approx) == 4:
        # 畫出梯形
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)

        # 印出四個點的座標
        for i, point in enumerate(approx):
            x, y = point[0]
            print(f"Point {i + 1}: ({x}, {y})")
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

# 顯示結果
cv2.imshow('Detected Rectangles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
