# auto_project
project/
├── main.py                        # 主控流程
├── tflite_segmentation.py        # SegmentationModel 類別
├── perspective_utils.py          # 做透視變換（Homography）
├── utils.py                      # mask 上色、找座標、畫框等工具
├── model.tflite                  # 模型檔
└── calibration/
    ├── camera_matrix.npy          
    ├── dist_coeffs.npy
    ├── rvecs.npy
    ├── tvecs.npy
    ├── homography.npy                  # 儲存好的 Homography 參數
└── 

# Project Folder Structure
1. images/:
Contains input chessboard images (JPEG/PNG) used for calibration.

3. model/:
train/
inference/

4. model_train/:
put the results of trained model 

5. else/:
stores some trivials code

# Camera Calibration
1. Use a **chessboard pattern**, e.g., 9×6 inner corners.
2. RMS reprojection error: 0.04 pixels (it's good!) then you will get some .npy files in ./calibration/
run
```
python camera_calibration.py
```
# Perspective Correction(transform)

run
```
python 
```


# 
