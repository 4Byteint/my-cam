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
POINTS = [[215.632, 126.0], [457.571, 126.0], [492.579, 415.726], [170.611, 409.155]] # 框偵測的四個點 
PERSPECTIVE_MATRIX_PATH = "calibration/perspective_matrix_180x220.npy"
PERSPECTIVE_SIZE = (180, 220) # (width, height)

# 模型輸入大小
MODEL_INPUT_SIZE = (220, 180) # (height, width)

# 分割後的角度分析
MIN_REGION_AREA = 50
Y_AXIS = [0, 1]


# 先上下排序 再左右排
SQUARE_POINTS = [[[21.377, 37.377], [71.901, 38.099], [71.477, 89.371], [21.642, 89.202]], [[111.737, 38.165], [161.718, 38.109], [161.327, 89.29], [111.379, 88.794]], [[21.596, 128.22], [71.471, 129.031], [71.525, 180.668], [21.265, 180.623]], [[111.661, 129.2], [160.97, 129.107], [160.816, 180.481], [111.471, 180.825]]] # 偵測到的正方形角點

# 右上-左上-左下-右下
CALIB_CIRCLES_PTS = [(136.2, 66.6), (48.6, 67.8), (43.8, 159.0), (137.4, 156.6)]
WORLD_CIRCLES_PTS = [(4.5, 38.7), (-4.5, 38.7), (-4.5, 29.7), (4.5, 29.7)]