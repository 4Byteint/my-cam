import math
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 載入新的圖像
new_image_path = "C:/Jill/Code/camera/model_train/predict_final/img71_predict_connector.png"
img = cv2.imread(new_image_path)

# 轉為灰階
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 邊緣檢測
edges = cv2.Canny(gray, 50, 150)

# 找輪廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找最大輪廓
max_cnt = max(contours, key=cv2.contourArea)

# 使用多邊形擬合
epsilon = 0.01 * cv2.arcLength(max_cnt, True)
approx = cv2.approxPolyDP(max_cnt, epsilon, True)

# 繪製所有邊
for i in range(len(approx)):
    pt1 = tuple(approx[i][0])
    pt2 = tuple(approx[(i+1)%len(approx)][0])
    cv2.line(img, pt1, pt2, (0, 255, 0), 2)

# 顯示結果
cv2.imshow("Detected Edges", img)
cv2.waitKey(0)
cv2.destroyAllWindows()