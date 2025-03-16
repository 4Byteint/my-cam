import cv2
import numpy as np
# 讀取影像
image = cv2.imread("./calibration/fixed_cam/img0.png")

# 棋盤格的尺寸（行數 x 列數）
rows, cols = 6, 9  # 內部角點數量

# 目標變換後的格子大小（假設每個格子 50x50 像素）
square_size = 50

# 計算變換後的影像尺寸
width = square_size * (cols - 1)
height = square_size * (rows - 1)

# 偵測棋盤格角點
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (cols - 1, rows - 1), None)

if ret:
    # 取得棋盤格四個角點（左上、右上、右下、左下）
    pts_src = np.array([
        corners[0][0],                     # 左上
        corners[cols - 2][0],              # 右上
        corners[-1][0],                    # 右下
        corners[-(cols - 1)][0]            # 左下
    ], dtype=np.float32)

    # 設定變換後的四個角對應位置
    pts_dst = np.array([
        [0, 0], [width, 0], [width, height], [0, height]
    ], dtype=np.float32)

    # 計算 Homography 矩陣
    H, _ = cv2.findHomography(pts_src, pts_dst)

    # 進行透視變換
    corrected_img = cv2.warpPerspective(image, H, (width, height))

    # 顯示結果
    cv2.imshow("Corrected Image", corrected_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("未找到棋盤格角點，請確保棋盤格清晰可見！")
