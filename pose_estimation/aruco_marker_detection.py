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

# ---------- 3. 方便把 rvec 轉 Euler 角（ZYX：Yaw→Pitch→Roll） ----------
def rvec_to_ypr(rvec):
    R, _ = cv2.Rodrigues(rvec)
    yaw   = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    pitch = np.degrees(np.arctan2(-R[2, 0],
                                  np.sqrt(R[2, 1]**2 + R[2, 2]**2)))
    roll  = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return yaw, pitch, roll       


 
# 初始化攝像頭
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
# before while True:
ret, frame = cap.read()
if not ret:
    raise RuntimeError("無法讀取第一張影像，請檢查攝影機")

h, w = frame.shape[:2]              # 取得高、寬
image_size = (w, h)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, image_size, 0, image_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
   
    # 轉換為灰度圖像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # 設置 ArUco 字典和檢測參數
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
 
    # 檢測 ArUco 標記
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
 
    if ids is not None:
        # 估計標記姿態
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.0552, newcameramtx, dist)
 
        # 繪製標記和坐標軸
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(frame, newcameramtx, dist, rvec, tvec, 0.03)
            # ② 算中心點 (X, Y, Z) 及 (Yaw, Pitch, Roll)
            x3d, y3d, z3d = tvec.flatten()
            yaw, pitch, roll = rvec_to_ypr(rvec)

            # ③ 終端印出
            print(f"X={x3d:.3f} m  Y={y3d:.3f} m  Z={z3d:.3f} m  "
                  f"Yaw={yaw:+6.1f}°  Pitch={pitch:+6.1f}°  Roll={roll:+6.1f}°")  ### ← 新增

            # ④ (可選) 疊文字到影像左上
            text = f"X:{x3d:.2f} Y:{y3d:.2f} Z:{z3d:.2f}"
            cv2.putText(frame, text, (10, 60), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            text_ang = f"Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}"
            cv2.putText(frame, text_ang, (10, 80), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
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

