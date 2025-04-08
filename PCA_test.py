import math
from PIL import Image
import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 載入新的圖像（Gradient Magnitude）
# new_image_path = "gradient_visualization_grip_new_img3_magnitude.png"
new_image_path = "C:/Jill/Thesis/Pic/213727.png"

new_image = Image.open(new_image_path).convert("RGB")
new_image_np = np.array(new_image)

# 轉為灰階
new_gray = cv2.cvtColor(new_image_np, cv2.COLOR_RGB2GRAY)

# 二值化處理，閾值可視狀況調整
_, new_binary = cv2.threshold(new_gray, 70, 255, cv2.THRESH_BINARY)
cv2.imshow("img", new_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 找出邊緣點
new_points = np.column_stack(np.where(new_binary > 0))

# 執行 PCA
new_pca = PCA(n_components=2)
new_pca.fit(new_points)
new_center = np.mean(new_points, axis=0)
new_axes = new_pca.components_
new_variance = new_pca.explained_variance_

# 計算主軸角度
new_principal_vector = new_axes[0]
new_angle_rad = math.atan2(new_principal_vector[0], new_principal_vector[1])
new_angle_deg = np.degrees(new_angle_rad)

# 輸出結果
new_pca_result = {
    "中心點座標 (y, x)": new_center,
    "第一主軸向量": new_principal_vector,
    "第二主軸向量": new_axes[1],
    "第一主軸變異值": new_variance[0],
    "第二主軸變異值": new_variance[1],
    "主軸方向角度 (度)": new_angle_deg
}

print(new_pca_result)

# 繪製主軸
plt.figure(figsize=(18, 18))
plt.imshow(new_image_np)
plt.scatter(new_center[1], new_center[0], color='red', s=100, label='center')

# 繪製第一主軸（紅色）
scale = 100  # 可以調整這個值來改變線的長度
plt.plot([new_center[1] - scale * new_axes[0][1], new_center[1] + scale * new_axes[0][1]],
         [new_center[0] - scale * new_axes[0][0], new_center[0] + scale * new_axes[0][0]],
         'r-', linewidth=2, label='first principle axis')

# 繪製第二主軸（藍色）
plt.plot([new_center[1] - scale * new_axes[1][1], new_center[1] + scale * new_axes[1][1]],
         [new_center[0] - scale * new_axes[1][0], new_center[0] + scale * new_axes[1][0]],
         'b-', linewidth=2, label='second principle axis')

plt.axis('off')
plt.legend()
plt.savefig('pca_result_1.png', bbox_inches='tight', pad_inches=0)
plt.close()

