# config.py
# picam2 setup
RESOLUTION = (640, 480)
VIDEO_FORMAT = "YUV420"
AF_MODE = 0
LENS_POSITION = 1.0
AWB_ENABLE = False
COLOUR_GAINS = (1.7, 0.7)
EXPOSURE_VALUE = -0.5

# camera calibration
CAMERA_MATRIX_PATH = "calibration/camera_matrix.npy"
DIST_COEFF_PATH = "calibration/dist_coeff.npy"

# 模型相關與轉換
MODEL_INPUT_SIZE = (220, 180) # (height, width)
PTH_MODEL_PATH = "./model_train/2025-06-21_16-04-17/unet-epoch463-lr0.0001.pth"
DEVICE = "cuda"
# ONNX_MODEL_PATH = "./model_train/2025-06-09_15-12-02/unet-epoch300-lr0.0001.onnx"
# TFLITE_MODEL_PATH = "./model_train/2025-06-09_15-12-02/"  # TFLite 模型路徑
# TFLITE_MODEL_NAME = "./model_train/2025-06-09_15-12-02/unet-epoch300-lr0.tflite"

# 影像轉換設定
POINTS = [(120, 0), (506, 0), (458, 366), (197, 369)] # 框偵測的四個點 
PERSPECTIVE_MATRIX_PATH = "calibration/perspective_matrix_180x220.npy"
PERSPECTIVE_SIZE = (180, 220) # (w, h)
# 座標轉成世界座標
HOMOGRAPHY_MATRIX_PATH = "calibration/homography_180x220.npy"

# 分割後的角度分析
MIN_REGION_AREA = 50
Y_AXIS = [0, 1]
