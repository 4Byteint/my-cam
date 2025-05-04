import time
import config
import cv2
import numpy as np
from picamera2 import Picamera2
import torch

from transform_application import apply_perspective_transform
from inference_segmentation import load_unet_model, predict_mask
from PCA import analyze_orientation
from utils.draw import draw_arrow  # 如果你要畫方向向量


# camera setup
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": config.RESOLUTION, "format": config.VIDEO_FORMAT})  # 使用 YUV420 格式
picam2.configure(config)
picam2.set_controls({
    "AfMode": config.AF_MODE,
    "LensPosition": config.LENS_POSITION,
    "AwbEnable": config.AWB_ENABLE,
    "ColourGains": config.COLOUR_GAINS,
    "ExposureValue": config.EXPOSURE_VALUE
})
picam2.start()
# model setup
device = torch.device(config.DEVICE)
model = load_unet_model(config.MODEL_PATH)
model.to(device).eval()

# calibration
mtx = np.load(config.CAMERA_MATRIX_PATH)
dist = np.load(config.DIST_COEFF_PATH)

# FPS

while True:
    # === Step 1: 讀取相機畫面 ===
    frame = picam2.capture_array("main")
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
    h, w = frame.shape[:2]
    flipped_frame = cv2.flip(frame,0)
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0.9, (w, h))
    dst = cv2.undistort(flipped_frame, mtx, dist, None, newcameramtx)
    
    # 顯示即時原始畫面
    cv2.imshow("Live", dst)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):
        print("[INFO] 按下 a，開始推論...")

        # Step 2: 做透視轉換 + resize
        warped = apply_perspective_transform(dst, output_size=config.RESIZED_SIZE)

        # Step 3: 丟進模型預測
        mask = predict_mask(warped, model, device)

        # Step 4: 做 PCA 分析
        center, direction, angle_deg = analyze_orientation(mask)

        # Step 5: 顯示結果
        vis = draw_arrow(warped.copy(), center, direction)
        cv2.putText(vis, f"Angle: {angle_deg:.2f}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 顯示推論視窗
        cv2.imshow("Segmentation + Direction", vis)

    elif key == ord('q'):
        print("[INFO] 按下 q，退出程式")
        break

cv2.destroyAllWindows()
picam2.stop()
