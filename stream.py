import numpy as np
from picamera2 import Picamera2, Preview
import cv2
import os
import time

# 初始化相機
picam2 = Picamera2()

config = picam2.create_video_configuration(main={"size":(640,480), "format": "YUV420"})  # 使用 YUV420 格式
# 如果要用 AI 推論可以用 lores

picam2.configure(config)
picam2.set_controls({
    "AfMode": 0,
    "LensPosition": 1.0,
    #"AwbEnable": False,
    #"ColourGains": (1.7, 0.8),
    #"ExposureValue": -0.5
})
picam2.start()

def showRealtimeImage(frame_name):
    base_count = 1
    base_path = "./calibration/perspective"
    mtx = np.load('./calibration/camera_matrix.npy')
    dist = np.load('./calibration/dist_coeff.npy')
    
    # 初始化 FPS 計算變數
    start_time = time.time()
    frame_count = 0
    fps = 0
    
    while True:
        frame = picam2.capture_array("main") 
        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)  # YUV 轉 BGR 才能顯示
        h, w = frame.shape[:2]
        #request = picam2.capture_request()  # 這樣影像會經過 Raspberry Pi 內建校正
        #frame = request.make_array("main")  # 轉換為 NumPy 陣列
        #request.release()  # 釋放請
        # 求，避免佔用相機資源
        # 修正色彩空間（RGB -> BGR）
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        flipped_frame = cv2.flip(frame,0)
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0.9, (w, h))
        dst = cv2.undistort(flipped_frame, mtx, dist, None, newcameramtx)
        
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1:
            fps = frame_count 
            print(f"FPS: {fps}")
            frame_count = 0
            start_time = time.time()
        
        # 在畫面上顯示 FPS
        #cv2.putText(flipped_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(frame_name, dst)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('b'):
            img_name = os.path.join(base_path, f"img{base_count}_trans.png")
            cv2.imwrite(img_name, dst) # change when you use calibration
            base_count += 1
    cv2.destroyAllWindows()
    picam2.stop()

# ==============================================================
# main
showRealtimeImage("Picamera2 image")

