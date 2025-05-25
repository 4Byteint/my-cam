import os
import numpy as np
import cv2
import config

# 添加 OpenCV 類型提示以解決 linter 錯誤
from typing import Any
cv2: Any

def ROI(img, points):
    """
    計算 ROI 的邊界框
    :param img: 原始影像
    :param points: 原始影像的四個角點 (左上、右上、左下、右下)
    :return: 裁剪後的影像
    """
    pts = np.array([points])
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.polylines(mask, pts, 1, 255)    
    cv2.fillPoly(mask, pts, 255)
    dst = cv2.bitwise_and(img, img, mask=mask)
    bg = np.ones_like(img, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)  
    
    # 計算 ROI 的邊界框
    x, y, w, h = cv2.boundingRect(pts)
    # 有需要黑底的 roi 可以這樣寫
    # cropped_roi = dst[y:y+h, x:x+w]
    # 建立白色背景並應用 mask
    bg = np.ones_like(img, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst_white = bg + dst
    # 裁剪白色背景的 ROI
    cropped_dst_white = dst_white[y:y+h, x:x+w]
    print(x, y)
    return cropped_dst_white

def perspective_transform_and_resize(image, points, H):
    """
    計算透視變換後的影像大小，並調整偏移量，使變換後的影像不固定在 (0,0)
    :param image: 原始影像
    :param H: 透視變換矩陣
    :param points: 原始影像的四個角點 (左上、右上、左下、右下)
    :param output_size: 輸出影像大小 (width, height)
    :return: 變換後的影像
    """
    # 轉換點為齊次座標
    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2) 
    warped_image = cv2.warpPerspective(image, H, config.PERSPECTIVE_SIZE) # (width, height)
    return warped_image

if __name__ == "__main__":
    # 檢查並創建 transform 目錄
    transform_dir = "./calibration/demo/transform"
    if not os.path.exists(transform_dir):
        os.makedirs(transform_dir)
        print(f"已創建目錄：{transform_dir}")
    
    img = cv2.imread("./calibration/demo/img2.png")
    H = np.load(config.PERSPECTIVE_MATRIX_PATH).astype(np.float32)
    points = np.array(config.POINTS)  # 框偵測的四個點 
    warped_image = perspective_transform_and_resize(img, points, H)
    cv2.imshow("warped_image", warped_image)
    cv2.imwrite(os.path.join(transform_dir, "img1_points.png"), warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

