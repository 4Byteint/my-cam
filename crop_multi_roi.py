import cv2
import os
def crop_trapezoid_with_bounded_mask(image_path, points):
    """
    使用遮罩來保留梯形區域，其他部分變黑，且輸出影像大小為梯形的包圍矩形大小
    :param image_path: 影像檔案路徑
    :param points: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] 梯形四個點的座標
    :return: 只保留梯形區域的影像，大小與梯形包圍矩形一致
    """
    image = cv2.imread(image_path)

    if image is None:
        print(f"錯誤：無法讀取影像 {image_path}，請檢查檔案路徑")
        return None

    # 轉換為 NumPy 陣列
    pts = np.array(points, dtype=np.int32)

    # 計算梯形的包圍矩形 (bounding box)
    x, y, w, h = cv2.boundingRect(pts)  # (x, y) 為左上角座標，(w, h) 為寬高

    # 創建與梯形包圍矩形大小相同的黑色遮罩
    mask = np.zeros((h, w), dtype=np.uint8)

    # 調整梯形座標，使其適應新的遮罩大小
    pts_adjusted = pts - [x, y]  # 讓 (x, y) 為 (0,0)

    # 在遮罩上填充白色，表示我們想要保留的區域
    cv2.fillPoly(mask, [pts_adjusted], 255)

    # 裁剪原始影像，使大小與 bounding box 相同
    cropped_image = image[y:y+h, x:x+w]

    # 只保留梯形區域，其他部分變黑
    masked_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)

    return masked_image

# 定義輸入和輸出目錄
input_dir = "./imprint/al/"
# output_dir_base = "C:/Jill/Code/data/no_touch/al/S001/"
# output_dir_sample = "C:/Jill/Code/data/touch/al/S001/"
output_dir_base = "./imprint/al/cropped/"
output_dir_sample = "./imprint/al/cropped/"
# 確保輸出目錄存在
os.makedirs(output_dir_base, exist_ok=True)
os.makedirs(output_dir_sample, exist_ok=True)


# 遍歷輸入目錄中的所有圖像文件
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        # 讀取圖片
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        
        image_path = "./imprint/al/img1.png"  # 替換為你的影像檔案
        trapezoid_points = [(147, 0), (492, 0), (468, 360), (197, 360)]  # 替換為你的梯形座標
        cropped_image = crop_trapezoid_with_bounded_mask(image_path, trapezoid_points)
        
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
