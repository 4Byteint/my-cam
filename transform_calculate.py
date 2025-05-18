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
        hull = cv2.convexHull(points.reshape(-1, 1, 2), returnPoints=True)
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
    :return: 方形資訊列表，每個方形包含角點座標、中心點和邊長資訊，以及是否為正方形的判斷
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
            # 取得角點座標
            corners = approx.reshape(-1, 2).astype(np.float32)
            # Sub-pixel 精度優化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            # 確保角點順序：左上、右上、右下、左下
            corners = order_points(corners)
            
            # 計算邊長
            side_lengths = []
            for i in range(4):
                next_i = (i + 1) % 4
                length = np.sqrt(
                    (corners[next_i][0] - corners[i][0])**2 +
                    (corners[next_i][1] - corners[i][1])**2
                )
                side_lengths.append(length)
            
            # 判斷是否為正方形
            is_square_shape = is_square(side_lengths)
            
            # 計算中心點
            center = np.mean(corners, axis=0)  # corners 是 shape (4, 2)
            print(f"中心點: {center}")
            
            # 在圖像上繪製
            # 繪製方形輪廓
            contour_color = (0, 255, 0) if is_square_shape else (0, 0, 255)  # 正方形綠色，非正方形紅色
            cv2.drawContours(output_image, [approx], -1, contour_color, 2)
            
            # 繪製角點
            for i, corner in enumerate(corners):
                x, y = corner.astype(int)
                cv2.circle(output_image, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(output_image, str(i), (x - 10, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 繪製中心點
            center_point = (int(center[0]), int(center[1]))
            cv2.circle(output_image, center_point, 6, (0, 255, 255), -1)
            cv2.putText(output_image, "C", (center_point[0] + 10, center_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 儲存方形資訊
            square_info = {
                'corners': corners,
                'side_lengths': side_lengths,
                'center': center,  # 儲存當前方形的中心點
                'is_square': is_square_shape  # 儲存是否為正方形的判斷結果
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
            print(f"是否為正方形: {'是' if is_square_shape else '否'}")
    
    # 顯示結果
    cv2.imshow("Detected Squares", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 返回所有方形的邊長資訊和是否為正方形的判斷
    return [{'side_lengths': square['side_lengths'], 'is_square': square['is_square']} for square in squares]


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

def is_square(side_lengths, tolerance=5):
    """
    根據四個邊長判斷是否為正方形
    :param side_lengths: 包含四個邊長的列表
    :param tolerance: 容許的誤差範圍（像素）
    :return: True（是正方形）或 False（不是正方形）
    """
    if not side_lengths or len(side_lengths) != 4:
        return False

    # 取得四個邊長
    side1, side2, side3, side4 = side_lengths

    # 計算每對邊長之間的差異
    diff1 = abs(side1 - side2)
    diff2 = abs(side2 - side3)
    diff3 = abs(side3 - side4)
    diff4 = abs(side4 - side1)

    print(f"邊長差異：{diff1:.3f}, {diff2:.3f}, {diff3:.3f}, {diff4:.3f}")
    
    # 如果所有邊長差異都小於容許值，則為正方形
    return all(diff <= tolerance for diff in [diff1, diff2, diff3, diff4])

######################################
# 计算透视变换参数矩阵
img_path= './calibration/demo/img0.png'

# 定義透視變換的四個點
points = np.array([(156, 41), (510, 29), (461, 349), (211, 351)]) # 框偵測的四個點
img = cv2.imread(img_path)
# cropped_img = ROI(img, points)
sorted_points = detect_points_2(img)
print(sorted_points)

if sorted_points is not None:
    # 更新 config.py 中的 POINTS
    with open('config.py', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # 找到 POINTS 行並更新
    for i, line in enumerate(lines):
        if line.startswith('POINTS ='):
            # 將座標四捨五入到兩位小數
            points_list = [[round(x, 2), round(y, 2)] for x, y in sorted_points.tolist()]
            lines[i] = f'POINTS = {points_list} # 框偵測的四個點 \n'
            break
    
    # 寫回文件
    with open('config.py', 'w', encoding='utf-8') as file:
        file.writelines(lines)
    
    print("已更新 config.py 中的 POINTS")
    
    transformed_img, H = apply_perspective_transform(img, sorted_points)
    detected_squares = detect_square(transformed_img)

    # 檢查是否所有檢測到的形狀都是正方形
    if detected_squares:  # 確保有檢測到形狀
        if all(square['is_square'] for square in detected_squares):
            np.save("./calibration/perspective_matrix_180x220.npy", H)
            print("所有形狀都是正方形，已保存 perspective_matrix_180x220.npy")
        else:
            print("不是所有形狀都是正方形，未保存矩陣")
    else:
        print("未檢測到任何形狀")
else:
    print("未能檢測到四個角點")
