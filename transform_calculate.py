# -*- coding:utf-8 -*-
import glob
import itertools
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import config 

# 添加 OpenCV 類型提示以解決 linter 錯誤
from typing import Any
cv2: Any

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
    print(f"ROI 偏移量 (x, y): ({x}, {y})")
    print(f"ROI 尺寸 (寬, 高): ({w}, {h})")
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
def detect_points_2(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), sigmaX=1)
    # Step 1: Edge detection
    edges = cv2.Canny(blurred, 20, 80)
    cv2.imshow("edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Step 3: HoughLinesP
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=150,
                            minLineLength=100, maxLineGap=15)
    
    if lines is None:
        print("未檢測到任何線條")
        return None
    
    
    # 過濾靠近底邊的線條
    height = image.shape[0]
    width = image.shape[1]
    bottom_threshold = int(height * 0.8)
    filtered_lines = []
    
    # 使用整數座標
    for line in lines:
        x1, y1, x2, y2 = map(int, line[0])
        if y1 < bottom_threshold and y2 < bottom_threshold:
            filtered_lines.append([x1, y1, x2, y2])
    
    if len(filtered_lines) < 4:
        print("檢測到的線條數量不足")
        return None
    
    # 繪製線條
    line_image = image.copy()
    for i, line in enumerate(filtered_lines):
        x1, y1, x2, y2 = line
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("line_image", line_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 計算交點
    def compute_line_intersection(line1, line2, epsilon=1e-10):
        """計算兩條線段的延伸直線交點，不管是否在線段內"""
        x1, y1, x2, y2 = map(float, line1)
        x3, y3, x4, y4 = map(float, line2)

        denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if abs(denom) < epsilon:
            return None  # 平行或重合

        px_num = (x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)
        py_num = (x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)
        px = px_num / denom
        py = py_num / denom

        return (px, py)  # 無條件返回延長線交點

    # 計算所有線段的交點
    intersections = []
    height, width = image.shape[:2]

    for i in range(len(filtered_lines)):
        for j in range(i + 1, len(filtered_lines)):
            pt = compute_line_intersection(filtered_lines[i], filtered_lines[j])
            if pt is not None:
                px, py = pt
                if 0 <= px < width and 0 <= py < height:
                    intersections.append((px, py))

    # 畫出有效交點
    for pt in intersections:
        x, y = map(int, pt)
        cv2.circle(line_image, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("intersections", line_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(intersections) < 4:
        print(f"找到的交點數量不足: {len(intersections)}")
        return None
    
    # 將交點轉換為 numpy 陣列
    intersections = np.array(intersections, dtype=np.float32)
    
    # 找出外圍的四個點
    def find_corner_points(points):
        if len(points) < 4:
            return None

        # 使用凸包找出外圍點
        points = np.array(points, dtype=np.float32)
        hull = cv2.convexHull(points.reshape(-1, 1, 2))
        hull_points = hull.reshape(-1, 2)

        # 超過4點就篩選距離最遠的4個
        N = len(hull_points)
        if N > 4:
            dist_matrix = np.zeros((N, N), dtype=np.float32)
            for i in range(N):
                for j in range(N):
                    dist_matrix[i][j] = np.linalg.norm(hull_points[i] - hull_points[j])

            max_dist_sum = 0
            best_points = None
            
            for comb in itertools.combinations(hull_points, 4):
                total = 0
                for p1, p2 in itertools.combinations(comb, 2):
                    total += np.linalg.norm(p1 - p2)
                if total > max_dist_sum:
                    max_dist_sum = total
                    best_points = comb
            hull_points = np.array(best_points, dtype=np.float32)

        # Y 排序 → 分上下，然後 X 排序 → 左右
        sorted_by_y = hull_points[hull_points[:, 1].argsort()]
        top_points = sorted_by_y[:2] 
        bottom_points = sorted_by_y[2:] 

        top_sorted = top_points[top_points[:, 0].argsort()]
        bottom_sorted = bottom_points[bottom_points[:, 0].argsort()]

        return np.array([
            top_sorted[0],      # 左上
            top_sorted[1],      # 右上
            bottom_sorted[1],   # 右下
            bottom_sorted[0]    # 左下
        ], dtype=np.float32)

    
    # 假設 intersections 是 np.ndarray or list of (x, y)
    corner_points = find_corner_points(intersections)
    print("交點數量：", len(intersections))
    print("交點內容：", intersections)

    if corner_points is not None and len(corner_points) == 4:

        corner_image = image.copy()
        for i, pt in enumerate(corner_points):
            print(f"角點 {i}: (x={pt[0]:.2f}, y={pt[1]:.2f})")
            cv2.circle(corner_image, tuple(pt.astype(int)), 8, (0, 255, 0), -1)
            cv2.putText(corner_image, f"{i}", (int(pt[0] + 15), int(pt[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # 顯示一次圖像，包含所有角點
        cv2.imshow("Corner Points", corner_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return corner_points
    else:
        print("未能找到合適的四個角點")
        return None
    
def detect_square(image):
    """
    偵測圖像中的所有方形，並計算其特徵
    :param image: 輸入圖像
    :return: 方形資訊列表，每個方形包含角點座標和邊長資訊
    """
    # 複製原圖用於繪製
    output_image = image.copy()
    
    # 預處理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imshow("edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 尋找輪廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow("output_image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    squares = []  # 儲存所有方形的資訊
    
    for contour in contours:
        # 計算輪廓周長
        peri = cv2.arcLength(contour, True)
        # 近似多邊形
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # 檢查是否為四邊形
        if len(approx) == 4:
            # 計算面積，過濾太小的區域
            area = cv2.contourArea(approx)
            if area < 100:  # 可調整的閾值
                continue
                
            # 取得角點座標
            corners = approx.reshape(-1, 2)
            
            # 確保角點順序：左上、右上、右下、左下
            corners = order_points(corners)
            
            # 計算邊的中點
            midpoints = []
            for i in range(4):
                next_i = (i + 1) % 4
                mid_x = (corners[i][0] + corners[next_i][0]) // 2
                mid_y = (corners[i][1] + corners[next_i][1]) // 2
                midpoints.append((mid_x, mid_y))
            
            # 計算邊長
            side_lengths = []
            for i in range(4):
                next_i = (i + 1) % 4
                length = np.sqrt(
                    (corners[next_i][0] - corners[i][0])**2 +
                    (corners[next_i][1] - corners[i][1])**2
                )
                side_lengths.append(length)
            
            # 計算中點之間的距離
            midpoint_distances = []
            for i in range(4):
                next_i = (i + 2) % 4  # 對角線的中點
                distance = np.sqrt(
                    (midpoints[next_i][0] - midpoints[i][0])**2 +
                    (midpoints[next_i][1] - midpoints[i][1])**2
                )
                midpoint_distances.append(distance)
            
            # 在圖像上繪製
            # 繪製方形輪廓
            cv2.drawContours(output_image, [approx], -1, (0, 255, 0), 2)
            
            # 繪製角點
            for i, corner in enumerate(corners):
                x, y = corner.astype(int)
                cv2.circle(output_image, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(output_image, str(i), (x - 10, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 繪製中點
            for i, midpoint in enumerate(midpoints):
                x, y = midpoint
                cv2.circle(output_image, (x, y), 3, (255, 0, 0), -1)
            
            # 儲存方形資訊
            square_info = {
                'corners': corners,
                'midpoints': midpoints,
                'side_lengths': side_lengths,
                'midpoint_distances': midpoint_distances,
                'area': area
            }
            squares.append(square_info)
            
            # 輸出資訊
            print(f"\n方形 {len(squares)}:")
            print("角點座標:")
            for i, corner in enumerate(corners):
                print(f"  角點 {i}: ({corner[0]:.1f}, {corner[1]:.1f})")
            print("邊長:")
            for i, length in enumerate(side_lengths):
                print(f"  邊 {i}: {length:.1f}")
            print("中點之間的距離:")
            for i, distance in enumerate(midpoint_distances):
                print(f"  距離 {i}: {distance:.1f}")
    
    # 顯示結果
    cv2.imshow("Detected Squares", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return squares

def order_points(pts):
    """
    將四個點按照左上、右上、右下、左下的順序排列
    """
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # 計算各點 x+y 的和
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    
    # 計算各點 y-x 的差
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    
    return rect

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
    border_threshold = 5
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

def apply_perspective_transform(image, sorted_points):
    """
    對影像應用透視變換，將偵測到的四邊形轉換為標準矩形
    :param image: 原始影像
    :param sorted_points: 四個角點座標 (左上、右上、右下、左下)
    :return: 透視變換後的影像
    """
    # 轉換為 NumPy 陣列
    sorted_points = np.array(sorted_points, dtype=np.float32)

    # 定義變換後的四個點 (標準矩形)
    dst_points = np.array([
        [0, 0],                                # 左上角
        [config.PERSPECTIVE_SIZE[0]-1, 0],     # 右上角
        [config.PERSPECTIVE_SIZE[0]-1, config.PERSPECTIVE_SIZE[1]-1],  # 右下角
        [0, config.PERSPECTIVE_SIZE[1]-1],     # 左下角
    ], dtype=np.float32)
    
    # 計算透視變換矩陣
    H = cv2.getPerspectiveTransform(sorted_points, dst_points)

    # 應用透視變換
    warped_image = cv2.warpPerspective(image, H, config.PERSPECTIVE_SIZE)
    cv2.imshow("warped_image", warped_image)
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
    print(f"{side1:.3f}", f"{side2:.3f}", f"{side3:.3f}", f"{side4:.3f}")

    if (abs(side1-side3) < tolerance or 
        abs(side2-side4) < tolerance or
        abs(side1-side2) < tolerance or
        abs(side3-side4) < tolerance):
        return True
    else: return False

######################################
# 计算透视变换参数矩阵
img_path= './calibration/demo/img0.png'

# 定義透視變換的四個點
points = np.array([(156, 41), (510, 29), (461, 349), (211, 351)]) # 框偵測的四個點
img = cv2.imread(img_path)
# cropped_img = ROI(img, points)
sorted_points = detect_points_2(img)
# print(sorted_points)
# transformed_img, H = apply_perspective_transform(img, sorted_points)
# inner_square_pts = detect_square(transformed_img)

# if is_square(inner_square_pts):
#     # np.save("./calibration/perspective_matrix_128x160.npy", H)
#     print("save perspective_matrix_128x160.npy")
# else:
#     print("not square")
