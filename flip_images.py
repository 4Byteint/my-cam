import os
import cv2
import glob
import shutil

def flip_images_in_folder(input_folder, output_folder):
    """
    將指定資料夾中的所有圖片上下顛倒並存到新的資料夾
    
    Args:
        input_folder: 輸入圖片資料夾路徑
        output_folder: 輸出圖片資料夾路徑
    """
    # 確保輸出資料夾存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已創建輸出資料夾：{output_folder}")
    
    # 獲取所有圖片文件
    image_files = glob.glob(os.path.join(input_folder, "*.[pj][np][g]"))  # 支援 .png 和 .jpg
    image_files.extend(glob.glob(os.path.join(input_folder, "*.[jJ][pP][eE][gG]")))  # 支援 .jpeg
    
    if not image_files:
        print(f"在 {input_folder} 中未找到圖片文件")
        return
    
    print(f"找到 {len(image_files)} 個圖片文件")
    
    # 處理每張圖片
    for img_path in image_files:
        # 讀取圖片
        img = cv2.imread(img_path)
        if img is None:
            print(f"無法讀取圖片：{img_path}")
            continue
        
        # 上下顛倒圖片
        flipped_img = cv2.flip(img, 0)  # 0 表示上下顛倒
        
        # 生成輸出文件路徑
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_folder, filename)
        
        # 儲存圖片
        cv2.imwrite(output_path, flipped_img)
        print(f"已處理並儲存：{filename}")

if __name__ == "__main__":
    # 設定輸入和輸出資料夾
    input_folder = "./calibration/demo"  # 輸入資料夾
    output_folder = "./calibration/demo/flipped"  # 輸出資料夾
    
    # 處理圖片
    flip_images_in_folder(input_folder, output_folder)
    print("\n所有圖片處理完成！") 