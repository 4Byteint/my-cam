import math
from PIL import Image
import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 載入新的圖像（Gradient Magnitude）
# new_image_path = "gradient_visualization_grip_new_img3_magnitude.png"
new_image_path = "C:/Jill/Code/camera/model_train/predict/unet-epoch166-lr0.0001_predict_connector_img89.png"

new_image = Image.open(new_image_path).convert("RGB")
new_image_np = np.array(new_image)

# 轉為灰階
new_gray = cv2.cvtColor(new_image_np, cv2.COLOR_RGB2GRAY)

# 二值化處理，閾值可視狀況調整
_, new_binary = cv2.threshold(new_gray, 70, 255, cv2.THRESH_BINARY)
# 找出邊緣點
new_points = np.column_stack(np.where(new_binary > 0))

# 執行 PCA
new_pca = PCA(n_components=2)
new_pca.fit(new_points)
new_center = np.mean(new_points, axis=0)
new_axes = new_pca.components_
new_variance = new_pca.explained_variance_

# 計算與 y 軸夾角（銳角）
def minimal_angle_with_y_axis(v):
    v = v / np.linalg.norm(v)
    y_axis = np.array([1, 0])  # row 方向，才是影像中的 y 軸
    dot = np.dot(v, y_axis)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    angle_deg = np.degrees(theta)
    return min(angle_deg, 180 - angle_deg)

angle1 = minimal_angle_with_y_axis(new_axes[0])
angle2 = minimal_angle_with_y_axis(new_axes[1])

best_axis = new_axes[0] if angle1 < angle2 else new_axes[1]
best_angle = min(angle1, angle2)

print("最靠近 y 軸的主軸向量：", best_axis)
print("與 y 軸最小夾角（銳角）：", best_angle, "度")

# 顯示主軸（只有最靠近 y 軸的那一條）
plt.figure(figsize=(18, 18))
plt.imshow(new_image_np)
plt.scatter(new_center[1], new_center[0], color='red', s=100, label='center')

scale = 100
plt.plot([new_center[1] - scale * best_axis[1], new_center[1] + scale * best_axis[1]],
         [new_center[0] - scale * best_axis[0], new_center[0] + scale * best_axis[0]],
         'r-', linewidth=2, label='axis closest to y')

plt.axis('off')
plt.legend()
plt.show()
