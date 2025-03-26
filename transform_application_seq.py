import os
import numpy as np
import cv2
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

def apply_persepctive(image, points):
    h,w = image.shape[:2]
    H = np.load("perspective_matrix.npy").astype(np.float32)
    """
    計算透視變換後的影像大小，並調整偏移量，使變換後的影像不固定在 (0,0)
    :param image: 原始影像
    :param H: 透視變換矩陣
    :param points: 原始影像的四個角點 (左上、右上、左下、右下)
    :return: 變換後的影像
    """
    # 轉換點為齊次座標
    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2) 
    warped_image = cv2.warpPerspective(image, H, (200, 250))
    # 如果你想轉換回來影像的時候可以用
    # H_inv = np.linalg.inv(H_translated)
    # warped_image2 = cv2.warpPerspective(warped_image, H_inv, (w, h))
    return warped_image

def threshold_OTSU_method(src):
    image = np.array(src)
    cimage = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # 灰度图
    th, dst = cv2.threshold(cimage, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
    circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, 1, 40, param1=50, param2=47, minRadius=0)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))  # 取整
        for i in circles[0, :]:
            x, y, r = i[0], i[1], i[2]
            # 畫圓
            cv2.circle(image, (x, y), r, (0, 0, 255), 2)
            # 畫圓心
            cv2.circle(image, (x, y), 2, (255, 0, 0), 2)
            # 顯示半徑資訊
            cv2.putText(image, f"r={r}", (x - 30, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        print("No circles detected.")

    cv2.imshow("otsu_circles", image)



#####################################################################
input_folder = './imprint/al_RGB/'
output_folder = './imprint/al_RGB/transform/'
os.makedirs(output_folder, exist_ok=True)
points = np.array([(136, 0), (508, 0), (457, 345), (203, 348)]) # 框偵測的四個點

# 處理每一張圖片
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"無法讀取圖片：{img_path}")
            continue

        cropped_img = ROI(img, points)
        warped_image = apply_persepctive(cropped_img, points)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, warped_image)
        print(f"已儲存：{output_path}")

cv2.imshow("cropped_img", cropped_img)
cv2.imshow("warped_image", warped_image)

cv2.waitKey(0)
cv2.destroyAllWindows()