import os
import numpy as np
from coor.test_world_coor import backproject_to_camera

here = os.path.dirname(__file__)               # …/camera/pose_estimation
root = os.path.abspath(os.path.join(here, '..'))  # …/camera

# 不論 cwd 在哪，都能正確定位
H    = np.load(os.path.join(root, "calibration", "perspective_matrix_128x160.npy")).astype(np.float32)
mtx  = np.load(os.path.join(root, "calibration", "camera_matrix.npy"))
dist = np.load(os.path.join(root, "calibration", "dist_coeff.npy"))
rvec = np.load(os.path.join(root, "calibration", "rvecs.npy"))
tvec = np.load(os.path.join(root, "calibration", "tvecs.npy"))

# # 載入必要的矩陣
# H = np.load("./calibration/perspective_matrix_128x160.npy").astype(np.float32)
# mtx = np.load("./calibration/camera_matrix.npy")
# dist = np.load("./calibration/dist_coeff.npy")
# rvec = np.load("./calibration/rvecs.npy")
# tvec = np.load("./calibration/tvecs.npy")

def process_point(u_pix, v_pix):
    """
    處理單個點的座標轉換
    """
    ray_cam, real_point = backproject_to_camera(
        u_pix=u_pix,
        v_pix=v_pix,
        K=mtx,
        H=H,
        roi_offset=(156, 29) # roi後左上角的座標決定
    )
    return ray_cam, real_point

# 使用示例
if __name__ == "__main__":
    # 測試單個點
    ray_cam, real_point = process_point(72, 131)
    print("射線在相機座標系中的方向:", ray_cam)
    print("實際點座標:", real_point)
    
    # 批量處理多個點
    points = [(72, 131), (80, 140), (90, 150)]
    for u, v in points:
        ray_cam, real_point = process_point(u, v)
        print(f"\n處理點 ({u}, {v}):")
        print("射線方向:", ray_cam)
        print("實際點座標:", real_point) 