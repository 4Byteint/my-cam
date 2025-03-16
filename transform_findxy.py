
import cv2
import numpy as np

# 讀取圖像
image = cv2.imread('./transform/img0_transform.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 進行二值化處理
_, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

# 找輪廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(gray, contours, -1, (0, 255, 0), 2)
# 顯示結果
cv2.imshow('Contours', gray)
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
