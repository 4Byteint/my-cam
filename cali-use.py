import cv2 as cv
import numpy as np

# **載入之前保存的標定結果**
mtx = np.load('camera_matrix.npy')
dist = np.load('dist_coeff.npy')
# 讀取攝影機解析度 (假設為 640x480)
frame_width, frame_height = 640, 480
# 預計算去畸變映射表（最快校正方法）
new_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (frame_width, frame_height), 1, (frame_width, frame_height))
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, new_mtx, (frame_width, frame_height), cv.CV_32FC1)

cap = cv.VideoCapture(0)  # 開啟攝影機


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 映射快速校正
    undistorted = cv.remap(frame, mapx, mapy, cv.INTER_LINEAR)
    
    # 顯示結果
    cv.imshow("Original", frame)
    cv.imshow("Undistorted", undistorted)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
