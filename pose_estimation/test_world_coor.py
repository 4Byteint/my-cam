import numpy as np
import cv2


def backproject_to_camera(u_pix, v_pix, roi_offset, H, K):
    """
    (u, v) 逆透射變換 (影像座標系)
    (u, v) 加上 roi_offset -> 還原參考原點 (影像座標系)
    反內參 -> ray_camera，還原到相機坐標系，得到反射光線方向 (相機坐標系)
    鏡面反射逆運算-> 得到入射光線方向 (相機坐標系)
    """
    P = np.array([u_pix, v_pix, 1.0])
    H_inv = np.linalg.inv(H)
    p_warp = H_inv @ P
    p_warp /= p_warp[2] # 除以z軸分量
    u_warp, v_warp = p_warp[0], p_warp[1]
    # print("u_warp: ", u_warp)
    # print("v_warp: ", v_warp)
    x_offset, y_offset = roi_offset
    u = u_warp + x_offset
    v = v_warp + y_offset
    # print("u: ", u)
    # print("v: ", v)
    
    pixel_homog = np.array([u, v, 1.0])
    ray_camera = np.linalg.inv(K) @ pixel_homog
    ray_camera /= np.linalg.norm(ray_camera)
    
    def calculate_normal_vector(deg):
        """
        Args:
            deg (_type_): 角度和相機座標系-y軸的夾角
        """
        # v = np.array([0,-1,0])
        deg_y_axis = 90 - deg
        cos_theta = np.cos(np.deg2rad(deg_y_axis)) 
        ny = -cos_theta
        nx = 0
        nz = np.sqrt(1-nx**2-ny**2)
        n = np.array([nx,ny,nz])
        # print("n: ", n)
        # print("n的長度: ", np.linalg.norm(n))
        return n
    normal_vector = calculate_normal_vector(60)
    # 鏡面反射逆運算-> 得到入射光線方向 
    def reflect_point_over_mirror(v_reflected, n):
        """
        正: v_reflected = vin - 2 * np.dot(vin, n) * n
        反: vin = v_reflected - 2 * np.dot(v_reflected, n) * n
        因為反射和入射向量對鏡面是對稱的
        """
        return v_reflected - 2 * np.dot(v_reflected, n) * n
    def intersect_ray_with_y_plane(ray_direction, y_value, ray_origin=np.array([0.0, 0.0, 0.0])):
        """
        計算從相機中心出發的光線，與 y = y_value 的平面交點

        Args:
            ray_direction (np.ndarray): 入射光線方向 (3D單位向量) [v_x, v_y, v_z]
            y_value (float): 平面 y = 常數值
            ray_origin (np.ndarray): 光線起點 (默認是相機光心 (0,0,0))

        Returns:
            np.ndarray or None: 交點的3D座標 (如果存在)，若光線與平面平行則返回 None
        """

        # 提取光線分量
        v_x, v_y, v_z = ray_direction
        o_x, o_y, o_z = ray_origin

        # 檢查光線是否與平面平行
        if np.isclose(v_y, 0.0):
            print("光線與 y = constant 平面平行，無交點。")
            return None

        # 計算交點參數 lambda
        lam = (y_value - o_y) / v_y

        # 計算交點
        intersection_point = ray_origin + lam * ray_direction

        return intersection_point
        
    
        
        
    ray_camera = reflect_point_over_mirror(ray_camera, normal_vector)
    real_point = intersect_ray_with_y_plane(ray_camera, -20.83)
    
    return ray_camera, real_point

H = np.load("../calibration/perspective_matrix_128x160.npy").astype(np.float32)
mtx = np.load("../calibration/camera_matrix.npy")
dist = np.load("../calibration/dist_coeff.npy")
rvec = np.load("../calibration/rvecs.npy")
tvec = np.load("../calibration/tvecs.npy")
# 假設你有一個點在最終影像上是
ray_cam, real_point = backproject_to_camera(
    u_pix=72,
    v_pix=131,
    K=mtx,
    H=H,
    roi_offset=(120, 0)  # ROI 的左上角偏移
)
print("Ray in camera coordinates:", ray_cam)
print("Real point:", real_point)
