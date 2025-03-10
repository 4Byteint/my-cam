import cv2
import numpy as np

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
    H, W = img_shape[:2]
    X, Y = np.meshgrid(np.arange(W) , np.arange(H) )
  
    # 轉換座標，使 (x_c, y_c) 成為圓心
    Xc = X - cx
    Yc = Y - cy

    # 計算球體 Z 值，確保 Z > 0（半圓球區域）
    Z = np.sqrt(np.maximum(r**2 - Xc**2 - Yc**2, 0) + 1e-6)  # 避免 sqrt 負數
    # 建立遮罩：只選擇球體內部區域
    mask = (Xc**2 + Yc**2 < r**2) & (Z > 0)
    # 計算表面梯度，只對半圓球區域有效
    Gx = np.where(mask, Xc / Z, 0)  # 非半圓球區域設為 0
    Gy = np.where(mask, Yc / Z, 0)
    print("mask True 的數量:", np.count_nonzero(mask))
    print("Z min:", np.min(Z))
    print("Z max:", np.max(Z))
    print("X 範圍:", np.min(Xc), np.max(Xc))
    print("Y 範圍:", np.min(Yc), np.max(Yc))
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
def lookup_table_dict(cx,cy,r, sample_path, Gx, Gy):
    img = cv2.imread(sample_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape[:2]
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
    # ===== 顯示 Lookup Table 結果 =====
    
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

    

# 設定影像路徑
base_path = "./trasform_img0_base.png"  # 替換為你的壓痕影像
sample_path = "./trasform_img2.png" 

# 產生 RGB 對應梯度的數據集
dataset = create_rgb2gradient_dataset(base_path, sample_path)
