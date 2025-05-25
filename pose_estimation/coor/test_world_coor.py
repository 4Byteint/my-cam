import os
import numpy as np
import cv2

def draw_coordinates(img, points, color=(0, 255, 0)):
    """
    在圖片上標記座標點
    Args:
        img: 輸入圖片
        points: 要標記的點列表，每個點是 (x, y, label) 的元組
        color: 標記的顏色，預設為綠色
    """
    img_copy = img.copy()
    for x, y, label in points:
        # 畫點
        cv2.circle(img_copy, (int(x), int(y)), 3, color, -1)
        # 寫座標
        cv2.putText(img_copy, label, (int(x)+5, int(y)-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img_copy

def backproject_to_camera(u_pix, v_pix, H, K, original_img=None):
    """
    (u, v) 是透射變換後的座標
    1. 使用 H_inv 矩陣將變換後座標轉換為原始圖片座標
    2. 在原始圖片和變換後圖片上標記對應的點
    3. 進行後續的相機座標系轉換
    """
    print("\n=== 光線追蹤調試信息 ===")
    print(f"輸入透射變換後座標: u={u_pix}, v={v_pix}")
    
    # 光線起點設為 (0,0,0)
    ray_origin = np.array([0.0, 0.0, 0.0])
    print(f"光線起點: {ray_origin}")
    
    # 在原始圖片和變換後圖片上標記座標
    if original_img is not None:
        # 使用 H_inv 計算原始圖片中的對應座標
        P = np.array([u_pix, v_pix, 1.0])
        H_inv = np.linalg.inv(H)
        p_original = H_inv @ P  # 使用 H_inv 矩陣計算原始座標
        p_original /= p_original[2]  # 歸一化
        u_orig, v_orig = p_original[0], p_original[1]
        print(f"對應的原始圖片座標: u={u_orig:.3f}, v={v_orig:.3f}")
        
        # 標記原始圖片
        points = [(u_orig, v_orig, f"({u_orig:.1f},{v_orig:.1f})")]
        marked_original = draw_coordinates(original_img, points)
        cv2.imshow("Original Image with Coordinates", marked_original)
        
        # 進行透射變換
        warped_img = cv2.warpPerspective(original_img, H, (180, 220))
        
        # 標記變換後的圖片（直接使用輸入的座標）
        points = [(u_pix, v_pix, f"({u_pix},{v_pix})")]
        marked_warped = draw_coordinates(warped_img, points)
        cv2.imshow("Warped Image with Coordinates", marked_warped)
    
    
    # 使用原始座標進行後續計算
    pixel_homog = np.array([u_orig, v_orig, 1.0])
    ray_camera = np.linalg.inv(K) @ pixel_homog
    ray_camera /= np.linalg.norm(ray_camera)
    print(f"相機座標系中的光線方向: {ray_camera}")
    
    def calculate_normal_vector(deg):
        """
        Args:
            deg (_type_): 角度和相機座標系-y軸的夾角
        """
        # v = np.array([0,-1,0])
        deg_y_axis = 90 - deg
        cos_theta = np.cos(np.deg2rad(deg_y_axis))
        sin_theta = np.sin(np.deg2rad(deg_y_axis))
        nx = 0
        ny = cos_theta
        nz = sin_theta
        n = np.array([nx,ny,nz])
        print(f"鏡面法向量: {n}")
        return n
    
    normal_vector = calculate_normal_vector(60)
    # 鏡面反射逆運算-> 得到入射光線方向 
    def reflect_point_over_mirror(v_in, n):
        """
        正: v_reflected = vin - 2 * np.dot(vin, n) * n
        反: vin = v_reflected - 2 * np.dot(v_reflected, n) * n
        因為反射和入射向量對鏡面是對稱的
        """
        v_out = v_in - 2 * np.dot(v_in, n) * n
        print(f"反射前光線方向: {v_in}")
        print(f"反射後光線方向: {v_out}")
        return v_out
    
    def intersect_ray_with_y_plane(ray_direction, y_value, ray_origin):
        """
        計算從指定起點出發的光線，與 y = y_value 的平面交點

        Args:
            ray_direction (np.ndarray): 入射光線方向 (3D單位向量) [v_x, v_y, v_z]
            y_value (float): 平面 y = 常數值
            ray_origin (np.ndarray): 光線起點

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
        print(f"光線起點: {ray_origin}")
        print(f"光線方向: {ray_direction}")
        print(f"交點參數 lambda: {lam}")
        print(f"計算得到的交點: {intersection_point}")
        return intersection_point
        
    
        
        
    ray_camera = reflect_point_over_mirror(ray_camera, normal_vector)
    real_point = intersect_ray_with_y_plane(ray_camera, -22.54, ray_origin)
    print(f"與 y=-22.54 平面的交點: {real_point}")
    print("=== 調試信息結束 ===\n")
    
    return ray_camera, real_point

if __name__ == "__main__":
    here = os.path.dirname(__file__)               # …/camera/pose_estimation
    root = os.path.abspath(os.path.join(here, '../..'))  # …/camera

    # 載入圖片
    original_img = cv2.imread("/home/jillisme/Documents/code/my-cam/calibration/demo/img0.png")
    
    if original_img is None:
        print("警告：無法載入圖片，將不會顯示座標標記")
    
    # 不論 cwd 在哪，都能正確定位
    H    = np.load(os.path.join(root, "calibration", "perspective_matrix_180x220.npy")).astype(np.float32)
    mtx  = np.load(os.path.join(root, "calibration", "camera_matrix.npy"))
    dist = np.load(os.path.join(root, "calibration", "dist_coeff.npy"))
    rvec = np.load(os.path.join(root, "calibration", "rvecs.npy"))
    tvec = np.load(os.path.join(root, "calibration", "tvecs.npy"))

    # 測試點（使用變換後的座標）
    test_points = [
        (0, 0),      # 左上
        (0, 220),    # 左下
        (90, 110),   # 中心點
        (180, 0),    # 右上
        (180, 220),  # 右下
    ]

    print("\n=== 測試不同位置的點（變換後座標）===")
    for u, v in test_points:
        print(f"\n測試點 (u={u}, v={v}):")
        ray_cam, real_point = backproject_to_camera(
            u_pix=u,
            v_pix=v,
            K=mtx,
            H=H,
            original_img=original_img
        )
        print(f"最終世界座標: {real_point}")
        cv2.waitKey(0)  # 等待按鍵，以便查看每個點的標記
    
    cv2.destroyAllWindows()  # 關閉所有視窗
