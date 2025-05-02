# detect_aruco_video_no_imutils.py
# 用法
#   python detect_aruco_video_no_imutils.py

import cv2
import time
import sys
import numpy as np

# ------ 1. 載入相機內參與畸變參數 ------
mtx = np.load("C:/Jill/Code/camera/pose_estimation/webcam_calib/camera_matrix.npy")
dist   = np.load("C:/Jill/Code/camera/pose_estimation/webcam_calib/dist_coeff.npy")

# ------ 2. 初始化 ArUco 字典與檢測參數 ------
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)



import numpy as np
import time
import cv2
 
 
# 初始化攝像頭
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    # 獲取圖像尺寸
    h, w = frame.shape[:2]
 
    # 糾正鏡頭扭曲
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    frame = frame[y:y+h, x:x+w]
 
    # 轉換為灰度圖像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # 設置 ArUco 字典和檢測參數
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
 
    # 檢測 ArUco 標記
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
 
    if ids is not None:
        # 估計標記姿態
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
 
        # 繪製標記和坐標軸
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.03)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.putText(frame, "Id: " + str(ids.flatten()), (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No Ids", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
 
    # 顯示結果
    cv2.imshow("frame", frame)
 
    key = cv2.waitKey(1)
    if key == 27:  # 按 ESC 鍵退出
        break
    elif key == ord(' '):  # 按空格鍵保存圖像
        filename = f"{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
 
cap.release()
cv2.destroyAllWindows()

