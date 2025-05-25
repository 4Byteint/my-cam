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

# 影像轉換設定
POINTS = [[174.0, 63.0], [488.751, 63.0], [452.758, 352.0], [214.792, 352.872]] # 框偵測的四個點 
PERSPECTIVE_MATRIX_PATH = "calibration/perspective_matrix_180x220.npy"
PERSPECTIVE_SIZE = (180, 220) # (width, height)

# 模型輸入大小
MODEL_INPUT_SIZE = (220, 180) # (height, width)

# 分割後的角度分析
MIN_REGION_AREA = 50
Y_AXIS = [0, 1]


# 先上下排序 再左右排
SQUARE_POINTS = [[[19.01, 41.27], [68.32, 41.9], [68.27, 91.42], [18.69, 92.25]], [[104.62, 41.59], [155.28, 41.06], [155.39, 91.21], [105.07, 91.07]], [[18.69, 132.87], [68.96, 132.83], [68.59, 182.62], [18.19, 182.45]], [[104.82, 132.75], [155.28, 133.44], [154.9, 182.68], [105.48, 181.76]]] # 偵測到的正方形角點
