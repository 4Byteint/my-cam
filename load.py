import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt


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

def detect_sphere_imprint(base_path, sample_path, min_radius=25, max_radius=45):
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
    cv2.imshow("gray", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 進行高斯模糊，減少雜訊影響
    gray_blurred = cv2.GaussianBlur(gray, (9,9), 2)

    # 使用霍夫圓變換來偵測圓形壓痕
    # miniDist:
    # param1: canny thershold
    # param2: hough 累積的 thershold 越小檢測到的圓越多
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=15, param2=20, minRadius=min_radius, maxRadius=max_radius)
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

def compute_surface_gradients(cx, cy, r, img_shape):
    """
    計算半圓球區域的表面梯度 (Gx, Gy)，忽略背景
    :param cx, cy: 圓心座標
    :param r: 球體半徑（像素）
    :param img_shape: 影像尺寸 (H, W)
    :return: 梯度場 (Gx, Gy) 和球體遮罩 mask
    """
    print(r)
    H, W = img_shape[:2]
    X, Y = np.meshgrid(np.arange(W) , np.arange(H) )
  
    # 轉換座標，使 Xc, Yc 相對於圓心
    Xc = X - cx
    Yc = Y - cy
    # 建立遮罩，只取半球內的像素
    mask = (Xc**2 + Yc**2 < r**2)
    # 計算球體 Z 值，確保 Z > 0（半圓球區域）
    Z = np.sqrt(np.maximum(r**2 - Xc**2 - Yc**2, 0) + 1e-6)  # 避免 sqrt 負數
   
    # 計算表面梯度，只對半圓球區域有效
    Gx = np.where(mask, Xc / Z, 0)
    Gy = np.where(mask, Yc / Z, 0)
    Gx[Z < 1e-2] = 0 # 挑出在 Gx 中 Z 太小的設為 0，保證邊界上的 Gx, Gy 不會因為 Z ≈ 0 產生無窮大
    Gy[Z < 1e-2] = 0 # 避免邊界處的計算異常
    print("Z min:", np.min(Z))
    print("Z max:", np.max(Z))
    print("X 範圍:", np.min(X), np.max(X))
    print("Y 範圍:", np.min(Y), np.max(Y))
    print("Gx 範圍:", np.min(Gx), np.max(Gx))
    print("Gy 範圍:", np.min(Gy), np.max(Gy))
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.quiver(X[::5, ::5], Y[::5, ::5], Gx[::5, ::5], Gy[::5, ::5], scale=50, color="red")
    plt.title("Gradient Field")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.show()

    return Gx, Gy, mask
def lookup_table_dict(sample_path, Gx, Gy):
    img = cv2.imread(sample_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    X, Y = np.meshgrid(np.arange(W) , np.arange(H) )
    R = img[:, :, 0]
    G = img[:, :, 1]
    dtype = [('x', 'i4'), ('y', 'i4'), ('R', 'i4'), ('G', 'i4'), ('Gx', 'f4'), ('Gy', 'f4')]
    lut = np.zeros(H*W, dtype=dtype)
    # 填充 Lookup Table
    # 將所有數據填入 lookup table
    lut['x'] = X.flatten()
    lut['y'] = Y.flatten()
    lut['R'] = R.flatten()
    lut['G'] = G.flatten()
    lut['Gx'] = Gx.flatten()
    lut['Gy'] = Gy.flatten()
    return lut

def visualize_gradient_heatmap(Gx, Gy):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax[0].imshow(Gx, cmap='coolwarm')
    ax[0].set_title("Gx Heatmap")
    plt.colorbar(im1, ax=ax[0])
    im2 = ax[1].imshow(Gy, cmap='coolwarm')
    ax[1].set_title("Gy Heatmap")
    plt.colorbar(im2, ax=ax[1])
    plt.show()

def query_lut(lut, x_query, y_query):
    result = lut[(lut['x'] == x_query) & (lut['y'] == y_query)]
    if result.size > 0:
        print(f"查詢座標 ({x_query}, {y_query})")
        print(f"R: {result['R'][0]}")
        print(f"G: {result['G'][0]}")
        print(f"Gx: {result['Gx'][0]:.10f}")  # 顯示 10 位小數
        print(f"Gy: {result['Gy'][0]:.10f}")  # 顯示 10 位小數
    else:
        print("未找到對應座標")
def save_lut_csv(lut, filename="lookup_table.csv"):
    df = pd.DataFrame(lut)
    df.to_csv(filename, index=False)
    print(f"LUT 已儲存為 {filename}")


def create_rgb2gradient_dataset(base_path, sample_path):
    """
    讀取影像，過濾背景，只儲存半圓球的 RGB + 梯度數據
    """
    # 讀取影像
    img = cv2.imread(sample_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 偵測球體
    cx, cy, r = detect_sphere_imprint(base_path, sample_path)

    # # 計算梯度，只取半圓球內部
    Gx, Gy, mask = compute_surface_gradients(cx, cy, r, img.shape)

    # 建立 lookup table
    lut = lookup_table_dict(sample_path, Gx, Gy)
    #visualize_lut(lut)
    # 可視化
    visualize_gradient_heatmap(Gx, Gy)
    
    # 查詢
    x_query, y_query = 300, 260
    query_lut(lut, x_query, y_query)
    save_lut_csv(lut)

# 設定影像路徑
base_path = "./transform_img0_base.png"  # 替換為你的壓痕影像
sample_path = "./transform_img2.png" 

# 產生 RGB 對應梯度的數據集
dataset = create_rgb2gradient_dataset(base_path, sample_path)
