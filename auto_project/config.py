# config.py
# picam2 setup
RESOLUTION = (640, 480)
VIDEO_FORMAT = "YUV420"
AF_MODE = 0
LENS_POSITION = 1.0
AWB_ENABLE = False
COLOUR_GAINS = (1.7, 0.8)
EXPOSURE_VALUE = -0.5

# camera calibration
CAMERA_MATRIX_PATH = "calibration/camera_matrix.npy"
DIST_COEFF_PATH = "calibration/dist_coeff.npy"

# 模型相關
MODEL_PATH = "model.pth"
DEVICE = "cuda"
TFLITE_MODEL_PATH = "./model_train/tflite_model/unet-epoch234-lr0.tflite"  # TFLite 模型路徑

# 影像轉換設定
POINTS = [(120, 0), (506, 0), (458, 366), (197, 369)] # 框偵測的四個點 
PERSPECTIVE_MATRIX_PATH = "calibration/perspective_matrix_128x160.npy"
PERSPECTIVE_SIZE = (128, 160) # (height, width)

# 模型輸入大小
MODEL_INPUT_SIZE = (160, 128) # (height, width)

# 分割後的角度分析
MIN_REGION_AREA = 50
Y_AXIS = [0, 1]
