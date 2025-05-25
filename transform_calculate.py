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
    """
    使用霍夫直線偵測，將相近點分群，從每群中選出最接近圖片中心的點作為角點
    :param image: 輸入圖像
    :return: 四個角點座標，按照左上、右上、右下、左下的順序排列
    """
    # 複製原圖用於繪製
    output_image = image.copy()
    
    # 預處理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1)
    edges = cv2.Canny(blurred, 20, 80)
    
    # 顯示邊緣檢測結果
    cv2.imshow("edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 使用霍夫變換找直線
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=110,
                           minLineLength=100, maxLineGap=15)
    
    if lines is None:
        print("未檢測到任何線條")
        return None
    
    # 過濾靠近底邊的線條
    height, width = image.shape[:2]
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
    for line in filtered_lines:
        x1, y1, x2, y2 = line
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # 計算所有線段的交點
    def compute_line_intersection(line1, line2, epsilon=1e-10):
        """計算兩條線段的延伸直線交點"""
        x1, y1, x2, y2 = map(float, line1)
        x3, y3, x4, y4 = map(float, line2)

        denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if abs(denom) < epsilon:
            return None

        px_num = (x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)
        py_num = (x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)
        px = np.float64(px_num) / np.float64(denom)
        py = np.float64(py_num) / np.float64(denom)

        return (px, py)

    # 收集所有有效交點
    intersections = []
    bottom_threshold = int(height * 0.8)  # 使用與線條過濾相同的底邊閾值
    for i in range(len(filtered_lines)):
        for j in range(i + 1, len(filtered_lines)):
            pt = compute_line_intersection(filtered_lines[i], filtered_lines[j])
            if pt is not None:
                px, py = pt
                # 過濾掉太靠近底邊的交點
                if 0 <= px < width and 0 <= py < bottom_threshold:
                    intersections.append((px, py))
    
    if len(intersections) < 4:
        print(f"找到的有效交點數量不足: {len(intersections)}")
        return None
    
    # 將交點轉換為 numpy 陣列
    points = np.array(intersections, dtype=np.float32)
    
    def cluster_points(points, distance_threshold=70):
        """
        將相近的點分群
        :param points: 點座標陣列
        :param distance_threshold: 分群距離閾值
        :return: 點群列表，每個群組包含該群的點
        """
        if len(points) == 0:
            return []
            
        # 初始化群組
        clusters = []
        remaining_points = points.copy()
        
        while len(remaining_points) > 0:
            # 選擇第一個點作為當前群組的中心
            current_point = remaining_points[0]
            current_cluster = [current_point]
            remaining_points = remaining_points[1:]
            
            # 尋找與當前點距離小於閾值的點
            i = 0
            while i < len(remaining_points):
                if np.linalg.norm(remaining_points[i] - current_point) < distance_threshold:
                    current_cluster.append(remaining_points[i])
                    remaining_points = np.delete(remaining_points, i, axis=0)
                else:
                    i += 1
            
            clusters.append(np.array(current_cluster))
        
        return clusters
    
    def select_center_points(clusters, image_center):
        """
        從每個群組中選出最接近圖片中心的點
        :param clusters: 點群列表
        :param image_center: 圖片中心點座標
        :return: 選出的四個角點
        """
        if len(clusters) < 4:
            return None
            
        selected_points = []
        for cluster in clusters:
            # 計算群組中每個點到圖片中心的距離
            distances = np.linalg.norm(cluster - image_center, axis=1)
            # 選出距離最小的點
            closest_point = cluster[np.argmin(distances)]
            selected_points.append(closest_point)
        
        return np.array(selected_points, dtype=np.float32)
    
    # 計算圖片中心點
    image_center = np.array([width/2, height/2], dtype=np.float32)
    
    # 將點分群
    clusters = cluster_points(points)
    print(f"找到 {len(clusters)} 個點群")
    
    # 在圖像上顯示分群結果
    cluster_image = line_image.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    for i, cluster in enumerate(clusters):
        color = colors[i % len(colors)]
        # 計算群組的中心點用於標記數字
        cluster_center = np.mean(cluster, axis=0)
        # 繪製群組中的所有點
        for point in cluster:
            x, y = point.astype(int)
            cv2.circle(cluster_image, (x, y), 4, color, -1)
        # 在群組中心標記數字
        cv2.putText(cluster_image, str(i+1), 
                   (int(cluster_center[0]), int(cluster_center[1])),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    # 繪製圖片中心點
    cv2.circle(cluster_image, tuple(image_center.astype(int)), 6, (0, 255, 255), -1)
    cv2.putText(cluster_image, "C", (int(image_center[0] + 10), int(image_center[1])),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("Clustered Points", cluster_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 從每個群組中選出最接近中心的點
    selected_points = select_center_points(clusters, image_center)
    
    if selected_points is None or len(selected_points) != 4:
        print("未能找到合適的四個角點")
        return None
    
    # 使用 order_points 函數確保角點順序
    ordered_points = order_points(selected_points)
    
    # 使用 cornerSubPix 進行亞像素級別的角點精確定位
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    # 將點轉換為正確的格式
    corners = ordered_points.reshape(-1, 1, 2).astype(np.float32)
    # 使用 cornerSubPix 進行精確定位
    cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
    # 轉換回原來的格式
    ordered_points = corners.reshape(-1, 2)
    
    # 在圖像上繪製最終結果
    final_image = line_image.copy()
    # 繪製所有交點
    for pt in intersections:
        x, y = map(int, pt)
        cv2.circle(final_image, (x, y), 3, (0, 255, 0), -1)
    
    # 繪製選中的四個角點
    for i, pt in enumerate(ordered_points):
        x, y = pt.astype(int)
        cv2.circle(final_image, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(final_image, str(i), (x + 10, y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # 顯示結果
    cv2.imshow("Final Points", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("檢測到的角點座標：")
    for i, pt in enumerate(ordered_points):
        print(f"角點 {i}: ({pt[0]:.3f}, {pt[1]:.3f})")
    
    # 在圖像上顯示過濾後的交點
    filtered_points_image = line_image.copy()
    for pt in intersections:
        x, y = map(int, pt)
        cv2.circle(filtered_points_image, (x, y), 4, (0, 255, 0), -1)
    
    # 繪製底邊閾值線
    cv2.line(filtered_points_image, (0, bottom_threshold), 
             (width, bottom_threshold), (0, 0, 255), 2)
    cv2.putText(filtered_points_image, "Bottom Threshold", 
                (10, bottom_threshold - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Filtered Intersection Points", filtered_points_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return ordered_points

def detect_square(image):
    """
    偵測圖像中的所有矩形，並計算其特徵
    :param image: 輸入圖像
    :return: 矩形資訊列表，每個矩形包含角點座標、中心點和邊長資訊，以及是否為正方形的判斷
    """
    # 複製原圖用於繪製
    output_image = image.copy()
    
    # 預處理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    edges = cv2.Canny(blurred, 50, 130)
    cv2.imshow("edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 定義結構元素
    kernel = np.ones((5, 5), np.uint8)  # 5x5 的矩陣作為結構元素
    # 使用閉合操作填補斷裂的線段
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("closed", closed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 尋找輪廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 只保留矩形輪廓
    rectangular_contours = []
    for contour in contours:
        # 計算輪廓周長
        peri = cv2.arcLength(contour, True)
        # 近似多邊形
        approx = cv2.approxPolyDP(contour, 0.01* peri, True)
        
        # 檢查是否為四邊形
        if len(approx) == 4:
            # 計算輪廓的面積
            area = cv2.contourArea(contour)
            # 計算最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            # 計算最小外接矩形的面積
            rect_area = cv2.contourArea(box)
            
            # 如果輪廓面積與最小外接矩形面積的比值接近1，則認為是矩形
            if area / rect_area > 0.85:
                rectangular_contours.append(contour)
    
    # 繪製所有矩形輪廓
    contour_image = image.copy()
    cv2.drawContours(contour_image, rectangular_contours, -1, (0, 255, 0), 2)
    cv2.imshow("Rectangular Contours", contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 計算每個矩形輪廓的左上角點
    contour_info = []
    for contour in rectangular_contours:
        # 計算輪廓的邊界框
        x, y, w, h = cv2.boundingRect(contour)
        contour_info.append({
            'contour': contour,
            'top_left': (x, y),  # 左上角點
            'height': y  # 用於上到下排序
        })
    
    # 先按照上到下排序（使用y座標）
    contour_info.sort(key=lambda x: x['height'])
    
    # 將輪廓分組（按照y座標相近的歸為同一行）
    rows = []
    current_row = []
    y_threshold = 20  # 同一行的y座標差異閾值
    
    for info in contour_info:
        if not current_row:
            current_row.append(info)
        else:
            # 如果當前輪廓與當前行的第一個輪廓y座標相近，加入當前行
            if abs(info['height'] - current_row[0]['height']) < y_threshold:
                current_row.append(info)
            else:
                # 對當前行按照左到右排序
                current_row.sort(key=lambda x: x['top_left'][0])
                rows.append(current_row)
                current_row = [info]
    
    # 處理最後一行
    if current_row:
        current_row.sort(key=lambda x: x['top_left'][0])
        rows.append(current_row)
    
    # 合併所有行
    sorted_contours = []
    for row in rows:
        sorted_contours.extend([info['contour'] for info in row])
    
    # 繪製排序後的輪廓
    cv2.drawContours(output_image, sorted_contours, -1, (0, 255, 0), 2)
    cv2.imshow("Sorted Rectangular Contours", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    squares = []  # 儲存所有矩形的資訊
    
    # 處理排序後的輪廓
    for i, contour in enumerate(sorted_contours):
        # 計算輪廓周長
        peri = cv2.arcLength(contour, True)
        # 近似多邊形
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # 取得角點座標
        corners = approx.reshape(-1, 2).astype(np.float32)
        
        # 使用 cornerSubPix 進行亞像素級別的角點精確定位
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        # 將點轉換為正確的格式
        corners = corners.reshape(-1, 1, 2)
        cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        corners = corners.reshape(-1, 2)
        
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
        center = np.mean(corners, axis=0)
        print(f"中心點: {center}")
        
        # 在圖像上繪製
        # 繪製矩形輪廓
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
        
        # 儲存矩形資訊
        square_info = {
            'corners': corners,
            'side_lengths': side_lengths,
            'center': center,
            'is_square': is_square_shape
        }
        squares.append(square_info)
        
        # 輸出資訊
        print(f"\n矩形 {len(squares)}:")
        print("角點座標:")
        for i, corner in enumerate(corners):
            print(f"  角點 {i}: ({corner[0]:.3f}, {corner[1]:.3f})")
        print("邊長:")
        for i, length in enumerate(side_lengths):
            print(f"  邊 {i}: {length:.3f}")
        print(f"是否為正方形: {'是' if is_square_shape else '否'}")
        print("")
    
    # 顯示結果
    cv2.imshow("Detected Rectangles", output_image)
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
img_path= './calibration/demo/img5.png'

# 定義透視變換的四個點

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
            # 將座標四捨五入到三位小數
            points_list = [[round(x, 3), round(y, 3)] for x, y in sorted_points.tolist()]
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
            # 將所有正方形的角點存入config.py
            square_points_list = []
            for square in detected_squares:
                # 將角點四捨五入到兩位小數
                corners = [[round(x, 2), round(y, 2)] for x, y in square['corners'].tolist()]
                square_points_list.append(corners)
            
            # 更新config.py
            with open('config.py', 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            # 檢查是否已存在SQUARE_POINTS
            square_points_exists = False
            for i, line in enumerate(lines):
                if line.startswith('SQUARE_POINTS ='):
                    lines[i] = f'SQUARE_POINTS = {square_points_list} # 偵測到的正方形角點\n'
                    square_points_exists = True
                    break
            
            # 如果不存在，則在文件末尾添加
            if not square_points_exists:
                lines.append(f'\nSQUARE_POINTS = {square_points_list} # 偵測到的正方形角點\n')
            
            # 寫回文件
            with open('config.py', 'w', encoding='utf-8') as file:
                file.writelines(lines)
            
            print("已更新 config.py 中的 SQUARE_POINTS")
            print(f"檢測到 {len(square_points_list)} 個正方形的角點")
            
            np.save("./calibration/perspective_matrix_180x220.npy", H)
            print("所有形狀都是正方形，已保存 perspective_matrix_180x220.npy")
        else:
            print("不是所有形狀都是正方形，未保存矩陣")
    else:
        print("未檢測到任何形狀")
else:
    print("未能檢測到四個角點")
