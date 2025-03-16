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

    # 偵測棋盤格角點 (初步偵測)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if not ret:
        print("未偵測到棋盤格，請確認影像")
        return None

    # 設定亞像素級細化參數
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)  # (類型, 最大迭代次數, 精度閾值)
    
    # 使用 cornerSubPix 進一步細化角點
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # 取出角點座標
    corners = corners.squeeze()  # (N, 1, 2) -> (N, 2)

    # 計算 X 方向的 mpp
    num_cols = chessboard_size[0] - 1  # 內部角點數量
    x_dists = [
        np.linalg.norm(corners[i] - corners[i + 1])
        for i in range(num_cols)
    ]
    avg_x_dist = np.mean(x_dists)  # 計算像素距離
    mpp_x = square_size / avg_x_dist  # mm/pixel

    # 計算 Y 方向的 mpp
    num_rows = chessboard_size[1] - 1
    y_dists = [
        np.linalg.norm(corners[i] - corners[i + chessboard_size[0]])
        for i in range(0, num_rows * chessboard_size[0], chessboard_size[0])
    ]
    avg_y_dist = np.mean(y_dists)
    mpp_y = square_size / avg_y_dist

    print(f"mpp (X方向): {mpp_x:.6f} mm/pixel")
    print(f"mpp (Y方向): {mpp_y:.6f} mm/pixel")
    return (mpp_x + mpp_y) / 2  # 取平均值作為 mpp

def detect_sphere_imprint(base_path, sample_path, min_radius=30, max_radius=60):
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
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    cv2.imshow("gray_blurred",gray_blurred)
    # **加入 Otsu 二值化**
    _, binary = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)     # 如果大於 127 就等於 255，反之等於 0。
    cv2.imshow("binary",binary)
    # **加入形態學運算（膨脹 → 侵蝕）**
    # kernel = np.ones((13, 13), np.uint8)  # 定義 5x5 內核
    # binary = cv2.dilate(binary, kernel, iterations=1)  # 先膨脹，使白色區域擴展
    # cv2.imshow("dilate",binary)
    # binary = cv2.erode(binary, kernel, iterations=1)   # 再侵蝕，使邊界平滑
    # v2.imshow("erode",binary)
    # 使用霍夫圓變換來偵測圓形壓痕
    # 圆心距：170 圆心距小于此值的圆不检测，以减小计算量
    # canny阈值：图像二值化的参数，根据实际情况调整
    # 投票数：一个圆需要至少包含多少个点，才认为这是一个圆
    circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
                               param1=50, param2=10, minRadius=min_radius, maxRadius=max_radius)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
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
        return 0,0,0

def compute_surface_gradients(cx, cy, r, img_path):
    """
    計算半球狀壓痕的表面梯度 (Gx, Gy)
    :param radius: 半球的物理半徑
    :param img_size: 影像尺寸 (寬, 高)
    :return: 梯度場 (Gx, Gy)
    """
    img = cv2.imread(img_path)
    (height, width) = img.shape[:2]
    # 生成像素座標網格
    X, Y = np.meshgrid(np.arange(width) - cx, np.arange(height) - cy)

    # 計算球體的表面 Z 座標
    Z = np.sqrt(r**2 - X**2 - Y**2 + 1e-6)  # 避免 sqrt 負數

    # 建立遮罩：只選擇球體內部區域
    mask = (X**2 + Y**2 < r**2) & (Z > 0)

    # 計算表面梯度，只對半圓球區域有效
    Gx = np.where(mask, X / Z, 0)  # 非半圓球區域設為 0
    Gy = np.where(mask, Y / Z, 0)

    return Gx, Gy, mask

def create_rgb2gradient_dataset(camera_matrix, dist_coeffs,base_path,sample_path):
    # 計算 mpp
    mpp_value = get_mpp("./calibration/fixed_cam/img1.png",  
                    chessboard_size=(6, 9),     
                    square_size=4,
                    camera_matrix=camera_matrix,     
                    dist_coeffs=dist_coeffs)
    print(f"最終 mpp: {mpp_value:.6f} mm/pixel")

# ############################################################
# # 壓痕影像偵測

    cx, cy, r = detect_sphere_imprint(base_path, sample_path)

# # ############################################################
# 計算梯度
    Gx, Gy, mask = compute_surface_gradients(cx, cy, mpp_value, sample_path)
    # 建立數據集（RGB 值 + 梯度）
    img = cv2.imread(sample_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dataset = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mask[i, j]:  # 只儲存半圓球內部的數據
                dataset.append([img_rgb[i, j, 0], img_rgb[i, j, 1], img_rgb[i, j, 2], Gx[i, j], Gy[i, j]])

    dataset = np.array(dataset)
    # 儲存數據集
    np.save("gradient_dataset.npy", dataset)
    print("數據集已儲存：gradient_dataset.npy")
    return dataset


base_path = "./imprint/al/transform/img_base.png"  # 替換為你的壓痕影像
sample_path = "./imprint/al/transform/img1.png" 
dataset = create_rgb2gradient_dataset(base_path, sample_path)




# ############################################################
# # 使用很多個梯度圖做為訓練資料，丟進模型訓練
