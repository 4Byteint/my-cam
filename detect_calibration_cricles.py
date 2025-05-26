import os
import numpy as np
import cv2
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
    cropped_roi = dst[y:y+h, x:x+w]
    # 建立白色背景並應用 mask
    bg = np.ones_like(img, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst_white = bg + dst
    # 裁剪白色背景的 ROI
    cropped_dst_white = dst_white[y:y+h, x:x+w]

    return cropped_dst_white

def apply_perspective(image, points, H):
    """
    計算透視變換後的影像大小，並調整偏移量，使變換後的影像不固定在 (0,0)
    :param image: 原始影像
    :param H: 透視變換矩陣
    :param points: 原始影像的四個角點 (左上、右上、左下、右下)
    :return: 變換後的影像
    """
    # 轉換點為齊次座標
    points = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
    warped_image = cv2.warpPerspective(image, H, config.PERSPECTIVE_SIZE)
    return warped_image

def detect_circles(warped_diff_img, warped_color_img):
    """
    對相減後的圖片進行圓形檢測
    :param warped_diff_img: 經過透視變換的差異圖片
    :param warped_color_img: 經過透視變換的彩色原圖
    :return: 圓心座標列表和處理後的圖片
    """
    # cv2.imshow("enhanced_img", enhanced_img)
    gray_blurred = cv2.GaussianBlur(warped_diff_img, (5,5), 2)
    cv2.imshow("gray_blurred", gray_blurred)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
    # 使用霍夫圓變換來偵測圓形壓痕
    # miniDist: 
    # param1: canny thershold
    # param2: hough 累積的 thershold 越小檢測到的圓越多
    circles = cv2.HoughCircles(gray_blurred, 
                               cv2.HOUGH_GRADIENT, 
                               dp=1.2, 
                               minDist=100,
                               param1=10, 
                               param2=30, 
                               minRadius=5, 
                               maxRadius=30)
   
    # 使用彩色原圖作為輸出圖片
    output_img = warped_color_img.copy()
    circle_centers = []
    
    # # 圓形檢測
    # circles = cv2.HoughCircles(
    #     img_clahe, 
    #     cv2.HOUGH_GRADIENT, 
    #     dp=2,
    #     minDist=50,
    #     param1=20,
    #     param2=55,
    #     minRadius=5,
    #     maxRadius=25
    # )
    
    if circles is not None:
        # 將結果轉換為浮點數，保持精確度
        circles = circles[0]
        
        for idx, circle in enumerate(circles):
            x, y, r = circle
            
            # 直接使用 HoughCircles 返回的浮點數座標
            circle_centers.append((x, y))
            
            # 在彩色圖片上標記
            cv2.circle(output_img, (int(x), int(y)), int(r), (0, 255, 0), 2)  # 綠色圓形
            cv2.circle(output_img, (int(x), int(y)), 2, (0, 0, 255), 3)  # 紅色圓心
            cv2.putText(output_img, f"#{idx}", (int(x)-10, int(y)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            print(f"圓心 #{idx} 座標: ({x:.2f}, {y:.2f})")
    
    return circle_centers, output_img

if __name__ == "__main__":
    # 檢查並創建輸出目錄
    output_folder = "./calibration/demo/transform"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已創建目錄：{output_folder}")
    
    # 基準圖片
    base_img = cv2.imread("./calibration/demo/img11.png")
    if base_img is None:
        print("無法讀取基準圖片")
        exit()
    
    # 要處理的樣本圖片列表
    sample_images = [
        "./calibration/demo/img7.png",
        "./calibration/demo/img8.png",
        "./calibration/demo/img9.png",
        "./calibration/demo/img10.png"
    ]
    
    # 儲存所有檢測到的圓心座標
    all_circle_centers = []
    
    # 載入透視變換矩陣
    H = np.load(config.PERSPECTIVE_MATRIX_PATH).astype(np.float32)
    points = np.array(config.POINTS)
    
    # 轉換基準圖片為灰度圖
    base_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    
    # 處理每張樣本圖片
    for sample_path in sample_images:
        print(f"\n處理圖片：{sample_path}")
        sample_img = cv2.imread(sample_path)
        if sample_img is None:
            print(f"無法讀取圖片：{sample_path}")
            continue
        
        # 轉換為灰度圖並相減
        sample_gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
        diff_img = cv2.absdiff(base_gray, sample_gray)
        
        # 對差異圖片和彩色圖片進行透視變換
        warped_diff = apply_perspective(diff_img, points, H)
        warped_color = apply_perspective(sample_img, points, H)
        
        # 檢測圓形
        circle_centers, result_img = detect_circles(warped_diff, warped_color)
        
        if circle_centers:
            # 將檢測到的圓心座標添加到總列表中
            all_circle_centers.extend(circle_centers)
            print(f"在 {sample_path} 中檢測到 {len(circle_centers)} 個圓形")
        
        # 顯示結果
        cv2.imshow(f"Detected Circles - {os.path.basename(sample_path)}", result_img)
        cv2.waitKey(0)  # 顯示1秒
        cv2.destroyAllWindows()
    
    # 更新 config.py 中的 CALIB_CIRCLES_XY
    if all_circle_centers:
        with open('config.py', 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # 將座標轉換為 [(x1,y1), (x2,y2), ...] 的形式，並四捨五入到三位小數
        circles_list = []
        for x, y in all_circle_centers:
            # 將 np.float32 轉換為 Python float
            x_float = float(round(x, 3))
            y_float = float(round(y, 3))
            circles_list.append([x_float, y_float])
        
        # 檢查是否已存在 CALIB_CIRCLES_XY
        calib_circles_exists = False
        for i, line in enumerate(lines):
            if line.startswith('CALIB_CIRCLES_XY ='):
                lines[i] = f'CALIB_CIRCLES_XY = {circles_list} # 校準圓形中心點座標 [(x1,y1), (x2,y2), ...]\n'
                calib_circles_exists = True
                break
        
        # 如果不存在，則在文件末尾添加
        if not calib_circles_exists:
            lines.append(f'\nCALIB_CIRCLES_XY = {circles_list} # 校準圓形中心點座標 [(x1,y1), (x2,y2), ...]\n')
        
        # 寫回文件
        with open('config.py', 'w', encoding='utf-8') as file:
            file.writelines(lines)
        
        print("\n已更新 config.py 中的 CALIB_CIRCLES_XY")
        print(f"總共檢測到 {len(all_circle_centers)} 個圓形中心點")
        print(f"座標列表: {circles_list}")
    else:
        print("\n未檢測到任何圓形")

