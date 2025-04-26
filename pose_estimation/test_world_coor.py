import numpy as np
import cv2

def reflect_point_over_mirror_y_axis_based(point, mirror_to_y_angle_deg):
    # 鏡面與 y 軸夾角 θ，則與 x 軸夾角為 90° - θ
    mirror_to_x_angle = 90 - mirror_to_y_angle_deg
    normal_angle = mirror_to_x_angle + 90  # 鏡面法線與 x 軸夾角
    theta = np.radians(normal_angle)

    # 建立法向量
    normal = np.array([np.cos(theta), np.sin(theta)])
    normal = normal / np.linalg.norm(normal)

    # 鏡射公式：P - 2 * (P·n) * n
    return point - 2 * np.dot(point, normal) * normal

def backproject_to_camera(u_prime, v_prime, H, K, roi_offset):
    x_offset, y_offset = roi_offset
    # Step 1: 透視變換還原
    pt_homo = np.array([u_prime, v_prime, 1.0])
    pt_roi = np.linalg.inv(H) @ pt_homo
    pt_roi /= pt_roi[2]

    # Step 2: 加回裁切偏移
    u = pt_roi[0] + x_offset
    v = pt_roi[1] + y_offset

   
    # Step 3: 組回方向向量（z=1）
    pixel = np.array([u, v, 1.0])
    ray_camera = np.linalg.inv(K) @ pixel
    ray_camera /= np.linalg.norm(ray_camera)
    
    # Step:4: 鏡面校正回來
    ray_camera = reflect_point_over_mirror_y_axis_based(ray_camera, 60)
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
    roi_offset=(120, 0)  # ROI 的左上角偏移
)
print("Ray in camera coordinates:", ray_cam)
