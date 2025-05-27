import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. 讀取 RGB 圖像並轉換為灰階
def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 轉換為灰階
    return gray

# 2. 計算 Sobel 梯度場
def compute_gradients(gray):
    Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  # X方向梯度
    Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)  # Y方向梯度
    return Gx, Gy

# 3. 透過 Fast Poisson Solver 恢復深度
def poisson_solver_fft(gradient_x, gradient_y):
    """使用 FFT 方法快速解泊松方程，從梯度場恢復深度圖"""
    div_g = np.gradient(gradient_x, axis=1) + np.gradient(gradient_y, axis=0)

    h, w = div_g.shape
    u = np.fft.fftfreq(w).reshape(1, -1)
    v = np.fft.fftfreq(h).reshape(-1, 1)
    denom = (2 * np.pi * u) ** 2 + (2 * np.pi * v) ** 2
    denom[0, 0] = 1  # 避免除零

    fft_div_g = np.fft.fft2(div_g)
    fft_z = fft_div_g / denom  # 解拉普拉斯方程
    fft_z[0, 0] = 0  # 消除 DC 分量

    depth_map = np.fft.ifft2(fft_z).real
    return depth_map

# 4. 使用閥值提取接觸區域
def extract_contact_region(depth_map, threshold=0.5):
    norm_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())  # 正規化
    contact_mask = norm_depth < threshold  # 小於閾值視為接觸區域
    return contact_mask.astype(np.uint8)

# 5. PCA 分析接觸區域的主軸方向
def compute_pca_direction(mask):
    y_indices, x_indices = np.where(mask > 0)  # 提取接觸點
    if len(x_indices) < 2:  # 如果點數太少，無法計算 PCA
        return None, None, None

    # 組合為 (x, y) 坐標點
    points = np.column_stack((x_indices, y_indices))

    # 執行 PCA
    pca = PCA(n_components=2)
    pca.fit(points)
    mean = pca.mean_
    principal_axes = pca.components_

    return mean, principal_axes, points

# 6. 視覺化結果
def visualize_results(gray, depth_map, contact_mask, mean, principal_axes):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 顯示原始灰階圖
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title("Grayscale Image")

    # 顯示深度圖，並添加 colorbar
    im = axes[1].imshow(depth_map, cmap='jet')
    fig.colorbar(im, ax=axes[1])  # ✅ 修正：用 fig.colorbar() 並指定 ax
    axes[1].set_title("Depth Map")

    # 顯示接觸區域與主軸
    axes[2].imshow(contact_mask, cmap='gray')
    axes[2].set_title("Contact Region with PCA")

    # 畫出 PCA 主軸
    if mean is not None:
        for i in range(2):
            vec = principal_axes[i] * 50  # 放大箭頭長度
            axes[2].arrow(mean[0], mean[1], vec[0], vec[1], color='red', head_width=5, head_length=5)

    plt.show()

# 7. 主函數
def process_image(img, threshold=0.5):
    gray = grayscale(img)
    Gx, Gy = compute_gradients(gray)
    depth = poisson_solver_fft(Gx, Gy)
    contact_mask = extract_contact_region(depth, threshold)
    mean, principal_axes, _ = compute_pca_direction(contact_mask)
    visualize_results(gray, depth, contact_mask, mean, principal_axes)



# 測試影像
base_path = "/home/jillisme/Documents/code/my-cam/trans-processing/cropped/img0_baseline.png"
sample_path = "/home/jillisme/Documents/code/my-cam/trans-processing/cropped/img0.png"
base_img = cv2.imread(base_path)
sample_img = cv2.imread(sample_path)
diff = cv2.absdiff(base_img, sample_img)
process_image(diff, threshold=0.6)  # 替換為你的影像路徑
plt.show()