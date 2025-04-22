import math
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 載入新的圖像
new_image_path = "C:/Jill/Code/camera/model_train/predict_final/img72_predict_connector.png"
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
epsilon = 0.025 * cv2.arcLength(max_cnt, True)
approx = cv2.approxPolyDP(max_cnt, epsilon, True)
centers = []
# 儲存邊的對應關係
center_edge_map = {}
# 繪製所有邊
for i in range(len(approx)):
    pt1 = tuple(approx[i][0])
    pt2 = tuple(approx[(i+1)%len(approx)][0])
    #cv2.line(img, pt1, pt2, (0, 255, 0), 2)
    # 計算中點座標
    mid_x = (pt1[0] + pt2[0]) // 2
    mid_y = (pt1[1] + pt2[1]) // 2
    center = (mid_x, mid_y)
    # 排除條件一：Y 約等於 160（±10）
    if abs(mid_y - 160) <= 10:
        continue

    # 排除條件二：X 約等於 0（+10）
    if abs(mid_x - 0) <= 10:
        continue
    centers.append(center)
    center_edge_map[center] = (pt1, pt2)
    cv2.line(img, pt1, pt2, (0, 255, 0), 2)
    
# 找出 y 座標最小者
min_y_point = min(centers, key=lambda point: point[1])
cv2.circle(img, min_y_point, 5, (0, 0, 255), -1)

print("邊的中點座標")
for idx, (x,y) in enumerate(centers, start=1): 
    x = int(x)
    y = int(y)
    print(f"第 {idx} 條邊中心點: {x, y}")


# 找出該點對應的邊 (pt1, pt2)
pt1, pt2 = center_edge_map[min_y_point]

# 計算邊的 dx, dy
dx = pt2[0] - pt1[0]
dy = pt2[1] - pt1[1]

# 處理斜率為 0 或 ∞ 的特殊情況
if dx == 0:
    perp_dx = 25
    perp_dy = 0
elif dy == 0:
    perp_dx = 0
    perp_dy = 25
else:
    # 計算垂線方向向量
    length = 25
    norm = math.sqrt(dx**2 + dy**2)
    perp_dx = int(-dy / norm * length)
    perp_dy = int(dx / norm * length)

# 計算垂線端點
ptA = (min_y_point[0] + perp_dx, min_y_point[1] + perp_dy)
ptB = (min_y_point[0] - perp_dx, min_y_point[1] - perp_dy)

# 畫出垂線（藍色）
cv2.line(img, ptA, ptB, (255, 0, 0), 2)
import math

# 計算向量（ptB → ptA）
vx = ptB[0] - ptA[0]
vy = ptB[1] - ptA[1]

# 計算與正 x 軸的夾角（使用 atan2，能正負方向分辨）
angle_rad = math.atan2(vy, vx)  # 單位：弧度
angle_deg = math.degrees(angle_rad)  # 轉為角度

# 讓角度範圍在 0~360 度之間（可選）
if angle_deg < 0:
    angle_deg += 360

print(f"垂直線與正 x 軸的夾角為：{angle_deg:.2f} 度")

# 顯示結果
cv2.imshow("Detected Edges", img)
cv2.waitKey(0)
cv2.destroyAllWindows()