import os
import cv2
import numpy as np
import shutil

# 設定資料夾路徑
input_folder = './auto_project/dataset/v1/original_img'      # 原始圖片資料夾
base_image_path = './auto_project/dataset/v1/original_img/base.png'       # base圖片路徑
output_folder = './auto_project/dataset/v1/diff_img'    # 輸出資料夾

# 建立輸出資料夾（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 讀取base圖片
base_img = cv2.imread(base_image_path, cv2.IMREAD_COLOR)
if base_img is None:
    raise FileNotFoundError(f"找不到 base 圖片：{base_image_path}")

# 取得所有圖片檔案（排除base圖片）
image_files = [f for f in os.listdir(input_folder)
               if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f != os.path.basename(base_image_path)]

# 處理每一張圖片
for filename in sorted(image_files):
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"無法讀取圖片：{filename}")
        continue

    # 確保圖片尺寸一致
    if img.shape != base_img.shape:
        print(f"圖片尺寸不符：{filename}")
        continue

    # 影像相減
    diff = cv2.absdiff(img, base_img)

    # 儲存結果，檔名與原圖一致
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, diff)
    print(f"已儲存：{output_path}")

# 複製 input_folder 內所有 json 檔案到 output_folder
for file in os.listdir(input_folder):
    if file.lower().endswith('.json'):
        src = os.path.join(input_folder, file)
        dst = os.path.join(output_folder, file)
        shutil.copy2(src, dst)
        print(f"已複製 JSON：{dst}")