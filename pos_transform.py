import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取前後影像
img_before = cv2.imread('C:/Jill/Code/camera/calibration/perspective/fortestcode/transform/img6_circles.png')
img_after = cv2.imread('C:/Jill/Code/camera/calibration/perspective/fortestcode/transform/img5_circles.png')

# 轉為灰階
gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)

# 相減取絕對差異
diff = cv2.absdiff(gray_after, gray_before)

# 模糊處理 + 門檻化
blurred = cv2.GaussianBlur(diff, (9, 9), 1)
_, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
cv2.imshow('blurred', thresh)
# 偵測圓形（Hough Transform）
circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                           param1=10, param2=10, minRadius=5, maxRadius=30)


# 儲存圓資訊
circle_data = []
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        circle_data.append((x, y, r))


# 收集圓心
# circle_centers = []
# if circles is not None:
#     circles = np.round(circles[0, :]).astype("int")
#     for (x, y, r) in circles:
#         circle_centers.append((x, y))

# 定義順時針排序函數（從左上開始）
def sort_clockwise(points):
    if len(points) != 4:
        return points
    cx = np.mean([p[0] for p in points])
    cy = np.mean([p[1] for p in points])
    def angle_from_center(p):
        return np.arctan2(p[1] - cy, p[0] - cx)
    return sorted(points, key=angle_from_center)

sorted_circle_data = sort_clockwise(circle_data)

# 在圖片上標註
output_img = img_after.copy()
for i, (x, y, r) in enumerate(sorted_circle_data):
    cv2.circle(output_img, (x, y), r, (0, 255, 0), 2)
    cv2.putText(output_img, f"{i+1}", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# 顯示結果（RGB）
output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

cv2.imshow('output_img', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
