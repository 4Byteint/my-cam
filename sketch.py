# 棋盤格校正完之後，我們可以計算出每個像素對應到多少毫米，這個值稱為 mpp (mm per pixel)。

import cv2
import numpy as np

def get_mpp(image_path, chessboard_size, square_size, camera_matrix=None, dist_coeffs=None):
    """
    使用棋盤格標定計算 mpp
    :param image_path: 棋盤格影像
    :param chessboard_size: 棋盤格內部角點數量 (cols, rows)
    :param square_size: 每個方格的真實尺寸 (mm)
    :return: 計算出的 mpp (mm/pixel)
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 若提供了內部參數與畸變係數，先進行去畸變處理
    if camera_matrix is not None and dist_coeffs is not None:
        gray = cv2.undistort(gray, camera_matrix, dist_coeffs)
    
    # 偵測棋盤格角點
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if not ret:
        print("未偵測到棋盤格，請確認影像")
        return None

    # 準備世界座標
    obj_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32) # 建立 (cols*rows, 3) 的世界座標
    obj_points[:, :2] = np.mgrid[0:chessboard_size[1], 0:chessboard_size[0]].T.reshape(-1, 2) # 轉成 (cols*rows, 2) 並填入 x, y 座標
    obj_points *= square_size # 乘上每個方格的真實尺寸

    # 計算單應性矩陣 H
    H, _ = cv2.findHomography(corners, obj_points[:, :2]) # 只取 x, y 座標

    # 計算 mpp：取 H 矩陣的 x 軸與 y 軸縮放因子
    mpp_x = np.linalg.norm(H[:, 0]) / H[2, 2] # 從 linalg.norm 計算 X Y 軸的總長度
    mpp_y = np.linalg.norm(H[:, 1]) / H[2, 2]

    print(f"mpp (X方向): {mpp_x:.6f} mm/pixel")
    print(f"mpp (Y方向): {mpp_y:.6f} mm/pixel")
    return (mpp_x + mpp_y) / 2  # 取平均值作為 mpp

def detect_sphere_imprint(base_path, sample_path, min_radius=10, max_radius=100):
    """
    使用霍夫圓變換偵測球體壓痕
    :param image_path: 壓痕影像的路徑
    :param min_radius: 可調整的最小半徑（像素）
    :param max_radius: 可調整的最大半徑（像素）
    :return: 壓痕的圓心座標 (x, y) 和半徑 r
    """
    # 讀取影像並轉換為灰階
    base_img = cv2.imread(base_path)
    sample_img = cv2.imread(sample_path)
    diff = cv2.absdiff(base_img, sample_img)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # 進行高斯模糊，減少雜訊影響
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 1)

    # 使用霍夫圓變換來偵測圓形壓痕
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        best_circle = max(circles[0, :], key=lambda c: c[2])  # 取半徑最大的圓
        for circle in circles[0, :]:
            x, y, r = circle
            # 在原圖上畫出所有圓
            cv2.circle(sample_img, (x, y), r, (0, 255, 0), 2) # 綠色圓圈
            cv2.circle(sample_img, (x, y), 2, (0, 0, 255), 3)  
        # 在原圖上畫出最大的圓
        x, y, r = best_circle
        cv2.circle(sample_img, (x, y), r, (255, 0, 0), 2) # 藍色圓圈
        cv2.circle(sample_img, (x, y), 2, (0, 0, 255), 3)  
        print(f"偵測到圓形壓痕: 圓心 ({x}, {y})，半徑 {r} 像素")
        cv2.imshow("Detected Circles", sample_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return x, y, r
    else:
        print("未偵測到圓形壓痕，請確認影像品質")
        return None

def compute_hemisphere_gradients(cx, cy, radius, img_size):
    """
    計算半球狀壓痕的表面梯度 (Gx, Gy)
    :param radius: 半球的物理半徑
    :param img_size: 影像尺寸 (寬, 高)
    :return: 梯度場 (Gx, Gy)
    """
    width, height = img_size
    X, Y = np.meshgrid(np.arange(width) - cx, np.arange(height) - cy)

    # 計算壓痕區域
    mask = (X**2 + Y**2) < radius**2  # 只計算半球內部區域

    # 避免平方根負數，先限制最大值
    denom = np.sqrt(radius**2 - X**2 - Y**2 + 1e-6)
    Gx = np.where(mask, X / denom, 0)  # 只保留半球內部的梯度
    Gy = np.where(mask, Y / denom, 0)

    return Gx, Gy


# 計算 mpp
camera_matrix = np.load("./camera_matrix.npy")
dist_coeffs = np.load("./dist_coeff.npy")
mpp_value = get_mpp("./calibration/img5_cali.png",  
                    chessboard_size=(4, 4),     
                    square_size=4,
                    camera_matrix=camera_matrix,     
                    dist_coeffs=dist_coeffs)
print(f"最終 mpp: {mpp_value:.6f} mm/pixel")



# ############################################################
# # 壓痕影像偵測
base_path = "./imprint/al/cropped/img0_base.png"  # 替換為你的壓痕影像
sample_path = "./imprint/al/cropped/img1.png" 
imprint_info = detect_sphere_imprint(base_path, sample_path)

# # ############################################################
# # # 表面梯度場計算
# # # 設定半球半徑和影像尺寸
if imprint_info:
    radius = 27  # 物理半徑 (單位像素)
    img_size = (200, 200)  # 影像尺寸

    # 計算梯度
    Gx, Gy = compute_hemisphere_gradients(radius, img_size)
    import matplotlib.pyplot as plt
    # 可視化梯度場
    fig, ax = plt.subplots(figsize=(6,6))
    ax.quiver(Gx, Gy, scale=10, color='r')  # 梯度向量圖
    ax.set_title("半球壓痕的表面梯度場")
    plt.show()
    print("Gx:", Gx)
    print("Gy:", Gy)

# ############################################################
# # 使用很多個梯度圖做為訓練資料，丟進模型訓練
