import cv2
import numpy as np

# 讀取圖片
image = cv2.imread("./calibration/img14_cali.png")

# 定義四個已知點 (變形前的四個角落)
src_points = np.array([(147, 0), (492, 0), (468, 360), (197, 360)], dtype=np.float32)

# 計算寬度 (取上邊與下邊的平均長度)
width_top = np.linalg.norm(src_points[0] - src_points[1])  # 左上到右上
width_bottom = np.linalg.norm(src_points[2] - src_points[3])  # 右下到左下
width = int(max(width_top, width_bottom))  # 選擇較大者作為寬度

# 計算高度 (取左邊與右邊的平均長度)
height_left = np.linalg.norm(src_points[0] - src_points[3])  # 左上到左下
height_right = np.linalg.norm(src_points[1] - src_points[2])  # 右上到右下
height = int(max(height_left, height_right))  # 選擇較大者作為高度

# 定義變換後的矩形座標
dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

# 計算透視變換矩陣
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# 進行透視變換
warped_image = cv2.warpPerspective(image, matrix, (width, height))

# 顯示結果
cv2.imshow("Original Image", image)
cv2.imshow("Warped Image", warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
