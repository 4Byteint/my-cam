import cv2
import numpy as np
import os
import glob

def cal_perspective_params(img, points):
    height, width = img.shape[:2]

    # 定義目標變換後的四邊形點（標準化視角）
    width_top = np.linalg.norm(points[1] - points[0])  
    width_bottom = np.linalg.norm(points[2] - points[3])  
    width = max(int(width_top), int(width_bottom))  

    height_left = np.linalg.norm(points[3] - points[0])  
    height_right = np.linalg.norm(points[2] - points[1])  
    height = max(int(height_left), int(height_right))  

    src = np.float32(points)  # 原始影像中的四點
    dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])  # 變換後對應的四點

    # 計算單應性矩陣（Homography Matrix），用於透視變換
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC)

    # 進行透視變換
    warped_img = cv2.warpPerspective(img, H, (width, height))

    return H, warped_img

# 影像處理主程式
input_folder = './imprint/al_calib/'
output_folder = './imprint/al_calib/homo/'

# 確保輸出資料夾存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 取得所有 PNG 圖片檔案
image_paths = sorted(glob.glob(os.path.join(input_folder, '*.png')))

# 定義透視變換的四個點（須根據實際影像調整）
points = np.array([(127, 15), (495, 8), (438, 346), (189, 348)])

for img_path in image_paths:
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"讀取失敗: {img_path}")
        continue

    # 計算透視變換
    H, transform_img = cal_perspective_params(img, points)

    # 顯示原始圖像與變換後圖像
    cv2.imshow('Original Image', img)
    cv2.imshow('Transformed Image', transform_img)

    # 儲存結果
    output_path = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(output_path, transform_img)

    # 顯示 1 秒後切換到下一張
    cv2.waitKey(1000)

cv2.destroyAllWindows()
