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
    # print(points)
    # # 計算透視變換後的座標
    # transformed_points = cv2.perspectiveTransform(points, H)
    # print(transformed_points)
    # # 找出最小 x, y (確保影像不會被固定在 (0,0))
    # min_x = np.min(transformed_points[:, 0, 0]) # 第一點x
    # min_y = np.min(transformed_points[:, 0, 1]) # 第一點y，但是負數
 
    # print(f"min_x: {min_x},min_y: {min_y}")
    # # 計算新影像的尺寸
    # max_x = np.max(transformed_points[:, 0, 0]) # 第三點x
    # max_y = np.max(transformed_points[:, 0, 1]) # 第三點y
    # # 計算 min_x 和 min_y，確保影像不會有負數座標
    # offset_x = -min_x if min_x < 0 else 0
    # offset_y = -min_y if min_y < 0 else 0

    # new_width = int(max_x + offset_x)  # max_x 不變，只加 offset_x
    # new_height = int(max_y + offset_y)  # max_y 不變，只加 offset_y

    # print(f"new_width: {new_width}, new_height: {new_height}")
    #print(min_x, max_x, min_y, max_y)
    # **調整透視變換矩陣 H，使影像不固定在 (0,0)**
    # translation_matrix = np.array([
    #     [1, 0, 100],  # X 軸偏移
    #     [0, 1, offset_y],  # Y 軸偏移
    #     [0, 0, 1]
    # ], dtype=np.float32)

    # **將透視變換矩陣 H 與平移矩陣相乘**
    # H_translated = np.dot(translation_matrix, H)
    # 執行透視變換
    warped_image = cv2.warpPerspective(image, H, (400, 500))
    # 如果你想轉換回來影像的時候可以用
    # H_inv = np.linalg.inv(H_translated)
    # warped_image2 = cv2.warpPerspective(warped_image, H_inv, (w, h))
    return warped_image

def threshold_OTSU_method(src):
    image = np.array(src)
    cimage = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # 灰度图
    th, dst = cv2.threshold(cimage, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
    circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, 1, 40, param1=50, param2=47, minRadius=0)
    circles = np.uint16(np.around(circles))  # 取整
    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)  # 在原图上画圆，圆心，半径，颜色，线框
        cv2.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)  # 画圆心
    cv2.putText(image, "param1=50, param2=47", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.imshow("otsu_circles", image)



#####################################################################
img_path = "C:/Jill/Code/camera/imprint/al/img1.png"
#"F:/img1_transform.png"
img = cv2.imread(img_path)
points = np.array([(136, 0), (508, 0), (457, 345), (203, 348)]) # 框偵測的四個點
cropped_img= ROI(img, points)
warped_image = apply_persepctive(cropped_img, points)
cv2.imshow("cropped_img", cropped_img)
cv2.imshow("warped_image", warped_image)

cv2.waitKey(0)
cv2.destroyAllWindows()