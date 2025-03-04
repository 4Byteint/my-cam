import cv2
import os
# 讀取圖片
image_path = "C:/Jill/Code/camera/trans-processing/al-sample.png"

image = cv2.imread(image_path)

# 設定裁剪區域的座標 (x1, y1) 為左上角, (x2, y2) 為右下角
x1, y1 = 210, 0  # 左上角
x2, y2 = 462, 335  # 右下角

# 確保 x1, x2, y1, y2 順序正確
x1, x2 = min(x1, x2), max(x1, x2)
y1, y2 = min(y1, y2), max(y1, y2)

# 定義輸入和輸出目錄
input_dir = "C:/Jill/Code/camera/trans-processing/inference/2-RG"
# output_dir_base = "C:/Jill/Code/data/no_touch/al/S001/"
# output_dir_sample = "C:/Jill/Code/data/touch/al/S001/"
output_dir_base = "C:/Jill/Code/camera/trans-processing/inference/2-RG/cropped/"
output_dir_sample = "C:/Jill/Code/camera/trans-processing/inference/2-RG/cropped/"
# 確保輸出目錄存在
os.makedirs(output_dir_base, exist_ok=True)
os.makedirs(output_dir_sample, exist_ok=True)


# 遍歷輸入目錄中的所有圖像文件
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        # 讀取圖片
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        # 裁剪影像
        cropped_image = image[y1:y2, x1:x2]
        # 生成輸出文件名和路徑
        base_name = os.path.splitext(filename)[0]
        if "base" in base_name:
            output_filename = f"{base_name.split('-')[0]}_baseline.png"
            output_path = os.path.join(output_dir_base, output_filename)
        elif "sample" in base_name:
            output_filename = f"{base_name.split('-')[0]}.png"
            output_path = os.path.join(output_dir_sample, output_filename)
        else:
            continue  # 跳過不符合條件的文件

        # 存檔
        cv2.imwrite(output_path, cropped_image)
        print(f"Saved: {output_path}")
        
print("Batch processing complete!")
