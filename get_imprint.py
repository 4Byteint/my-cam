import numpy as np
from picamera2 import Picamera2, Preview
import cv2
import os

# 初始化相機
picam2 = Picamera2()

camera_config = picam2.create_still_configuration(main={"size":(640,480)})  # 使用預覽模式
picam2.configure(camera_config)
# 關閉自動對焦(Af)，設置為手動模式
picam2.set_controls({"AfMode": 0, "LensPosition": 1.0})  # 固定焦距到 1/10m = 10cm
picam2.set_controls({"AwbEnable": False, "ColourGains": (1.5, 1.0)})  # 1.7/0.7關掉白平衡，調整  Gain 值
picam2.set_controls({"ExposureValue": -0.5})  # +1 EV 提高亮度
picam2.start()
# 加載相機內部參數和畸變參數
camera_matrix = np.load("./camera_matrix.npy")
dist_coeffs = np.load("./dist_coeff.npy")

def showRealtimeImage(frame_name):
    base_count = 0
    base_path = "./imprint/al/"

    while True:
        frame = picam2.capture_array()
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        # 修正色彩空間（RGB -> BGR）
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        flipped_frame = cv2.flip(frame_bgr,0)
        cv2.imshow(frame_name, flipped_frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('b'):
            img_name = os.path.join(base_path, f"img{base_count}.png")
            cv2.imwrite(img_name, flipped_frame)
            base_count += 1

def getFrame(frame_name):
    frame = picam2.capture_array()
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
     # 修正色彩空間（RGB -> BGR）
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    flipped_frame = cv2.flip(frame_bgr,0)
    # 使用 OpenCV 顯示影像
    cv2.imwrite(frame_name, flipped_frame)


# ==============================================================
# main
showRealtimeImage("Picamera2 image")
picam2.stop()
