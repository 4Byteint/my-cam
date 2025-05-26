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
POINTS = [[170.054, 68.341], [492.274, 64.55], [457.564, 353.0], [215.164, 353.0]] # 框偵測的四個點 
PERSPECTIVE_MATRIX_PATH = "calibration/perspective_matrix_180x220.npy"
PERSPECTIVE_SIZE = (180, 220) # (width, height)

# 模型輸入大小
MODEL_INPUT_SIZE = (220, 180) # (height, width)

# 分割後的角度分析
MIN_REGION_AREA = 50
Y_AXIS = [0, 1]


# 先上下排序 再左右排

SQUARE_POINTS = [[[21.32, 38.999], [71.423, 38.564], [71.378, 90.248], [21.641, 91.32]], [[111.33, 38.097], [160.813, 38.123], [160.873, 89.729], [111.536, 89.806]], [[21.703, 130.232], [71.424, 129.865], [71.758, 180.948], [21.491, 181.808]], [[111.227, 130.185], [161.25, 129.582], [161.661, 180.89], [111.609, 180.888]]] # 偵測到的正方形角點

# 右下-左下-左上-右上
CALIB_CIRCLES_PTS = [[135.0, 154.1999969482422], [45.0, 154.1999969482422], [42.599998474121094, 55.79999923706055], [135.0, 63.0]] # 校準圓形中心點座標 [(x1,y1), (x2,y2), ...]
# real-world 
WORLD_CIRCLES_PTS = [[4.5, 26.7], [-4.5, 26.7], [-4.5, 35.7], [-4.5, 35.7]]