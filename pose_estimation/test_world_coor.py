import numpy as np
import cv2

def backproject_to_camera(u_prime, v_prime, H, K, distCoeffs, roi_offset):
    x_offset, y_offset = roi_offset

    # Step 1: 透視變換還原
    pt_homo = np.array([u_prime, v_prime, 1.0])
    pt_roi = np.linalg.inv(H) @ pt_homo
    pt_roi /= pt_roi[2]

    # Step 2: 加回裁切偏移
    u = pt_roi[0] + x_offset
    v = pt_roi[1] + y_offset

    # Step 3: 畸變校正
    undistorted = cv2.undistortPoints(
        np.array([[[u, v]]], dtype=np.float32),
        K,
        distCoeffs
    )
    # Step 4: 組回方向向量（z=1）
    x_c, y_c = undistorted[0, 0]
    ray_camera = np.array([x_c, y_c, 1.0])
    ray_camera /= np.linalg.norm(ray_camera)
    return ray_camera


H = np.load("../calibration/perspective_matrix_128x160.npy").astype(np.float32)
mtx = np.load("../calibration/camera_matrix.npy")
dist = np.load("../calibration/dist_coeff.npy")
# 假設你有一個點在最終影像上是 (150, 100)
ray_cam = backproject_to_camera(
    u_prime=72,
    v_prime=131,
    H=H,
    K=mtx,
    distCoeffs=dist,
    roi_offset=(120, 0)  # ROI 的左上角偏移
)
print("Ray in camera coordinates:", ray_cam)
