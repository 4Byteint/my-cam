import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time

# 檢查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 定義神經網絡結構（需要與訓練時相同）
class GradientNN(nn.Module):
    def __init__(self):
        super(GradientNN, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 2)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


def process_image(image_path):
    # 記錄開始時間
    start_time = time.time()
    
    # 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"無法讀取圖片: {image_path}")
    
    # 獲取圖片尺寸
    height, width = img.shape[:2]
    # 準備模型
    model = GradientNN()
    model.load_state_dict(torch.load("gradient_model_RGB_norm.pth"))
    model.to(device)
    model.eval()
    # 建立所有 pixel 的輸入資料
    inputs = []
    for y in range(height):
        for x in range(width):
            b, g, r = img[y, x]
            inputs.append([x, y, r, g, b])
    input_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
    # 3. 批量推論
    with torch.no_grad():
        outputs = model(input_tensor).cpu().numpy()
    # 創建梯度圖
    gradient_magnitude = np.zeros((height, width))
    gradient_direction = np.zeros((height, width))
    
    # 創建 Gx 和 Gy 的陣列
    gx_values = np.zeros((height, width))
    gy_values = np.zeros((height, width))
    
    idx = 0
    for y in range(height):
        for x in range(width):
            b, g, r = img[y, x]
            gx, gy = outputs[idx]
            gx_values[y, x] = gx
            gy_values[y, x] = gy
            gradient_magnitude[y, x] = np.sqrt(gx**2 + gy**2) 
            gradient_direction[y, x] = np.arctan2(gy, gx)
            idx += 1
    
    # 計算並打印 Gx 和 Gy 的範圍
    print(f"Gx 範圍: [{gx_values.min():.4f}, {gx_values.max():.4f}]")
    print(f"Gy 範圍: [{gy_values.min():.4f}, {gy_values.max():.4f}]")
    
    total_time = time.time() - start_time
    print(f"圖片處理總時間: {total_time:.2f} 秒")
    
    return gradient_magnitude, gradient_direction, img

def visualize_gradients(gradient_magnitude, gradient_direction, original_img, output_name="gradient_visualization"):
    """_summary_

    Args:
        gradient_magnitude (numpy.ndarray): 所有 pixel 的梯度大小
        gradient_direction (numpy.ndarray): 所有 pixel 的梯度方向
        original_img (numpy.ndarray): BGR 格式
        output_name (str): 輸出文件名（不包含擴展名）
    """
    start_time = time.time()
    
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    # 多圖顯示
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # 顯示原始圖片
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off') # 不顯示座標軸
    
    # 顯示梯度大小
    axes[0, 1].imshow(gradient_magnitude, cmap='gray') 
    axes[0, 1].set_title('Gradient Magnitude')
    axes[0, 1].axis('off')
    
    # 顯示梯度方向
    axes[1, 0].imshow(gradient_direction, cmap='hsv') 
    axes[1, 0].set_title('Gradient Direction')
    axes[1, 0].axis('off')
    
    # 顯示梯度向量場
    y, x = np.mgrid[0:gradient_magnitude.shape[0]:2, 0:gradient_magnitude.shape[1]:2]
    gx = np.cos(gradient_direction[::2, ::2]) * gradient_magnitude[::2, ::2]
    gy = -np.sin(gradient_direction[::2, ::2]) * gradient_magnitude[::2, ::2] # 梯度方向反向，讓影像和數學坐標系一致
    axes[1, 1].imshow(original_img)
    axes[1, 1].quiver(x, y, gx, gy, gradient_magnitude[::2, ::2], cmap='hot', alpha=0.6)
    axes[1, 1].set_title('Gradient Vector Field')
    axes[1, 1].axis('off')

    # 調整子圖之間的間距
    plt.tight_layout()
    
    # 保存組合圖
    plt.savefig(f"{output_name}.png", bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
    # 單獨保存梯度大小圖
    plt.figure(figsize=(10, 10))
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.axis('off')
    plt.savefig(f"{output_name}_magnitude.png", bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
    visualization_time = time.time() - start_time
    print(f"可視化時間: {visualization_time:.2f} 秒")

def analyze_gradient_directions(gx_values, gy_values, original_img, output_name="gradient_analysis"):
    """
    分析梯度方向，找出梯度方向差異大且兩側梯度大小相似的區域
    
    Args:
        gx_values (numpy.ndarray): Gx 梯度值
        gy_values (numpy.ndarray): Gy 梯度值
        original_img (numpy.ndarray): 原始圖片
        output_name (str): 輸出文件名前綴
    """
    height, width = gx_values.shape
    
    # 計算梯度大小和方向
    gradient_magnitude = np.sqrt(gx_values**2 + gy_values**2)
    directions = np.arctan2(gy_values, gx_values)
    
    # 計算梯度大小的閾值（使用梯度大小的均值作為參考）
    magnitude_threshold = np.mean(gradient_magnitude)
    
    # 創建一個掩碼來標記符合條件的區域
    gradient_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 設定鄰域大小和相似度閾值
    neighborhood_size = 5
    magnitude_similarity_threshold = 0.2  # 梯度大小相似度閾值（20%差異）
    
    # 遍歷每個像素（除了邊界）
    for y in range(neighborhood_size, height - neighborhood_size):
        for x in range(neighborhood_size, width - neighborhood_size):
            current_magnitude = gradient_magnitude[y, x]
            
            # 只考慮當前像素梯度大小超過閾值的情況
            if current_magnitude > magnitude_threshold:
                current_direction = directions[y, x]
                
                # 檢查鄰域內的像素
                for dy in range(-neighborhood_size, neighborhood_size + 1):
                    for dx in range(-neighborhood_size, neighborhood_size + 1):
                        if dx == 0 and dy == 0:
                            continue
                            
                        neighbor_y = y + dy
                        neighbor_x = x + dx
                        
                        neighbor_magnitude = gradient_magnitude[neighbor_y, neighbor_x]
                        
                        # 檢查梯度大小是否相似
                        magnitude_ratio = min(current_magnitude, neighbor_magnitude) / max(current_magnitude, neighbor_magnitude)
                        
                        if magnitude_ratio > (1 - magnitude_similarity_threshold):
                            # 計算兩個方向之間的夾角
                            neighbor_direction = directions[neighbor_y, neighbor_x]
                            angle_diff = abs(current_direction - neighbor_direction)
                            
                            # 如果夾角大於90度
                            if angle_diff > np.pi/2:
                                # 檢查是否在兩側都有相似大小的梯度
                                opposite_y = y - dy
                                opposite_x = x - dx
                                
                                if (0 <= opposite_y < height and 0 <= opposite_x < width):
                                    opposite_magnitude = gradient_magnitude[opposite_y, opposite_x]
                                    opposite_ratio = min(current_magnitude, opposite_magnitude) / max(current_magnitude, opposite_magnitude)
                                    
                                    if opposite_ratio > (1 - magnitude_similarity_threshold):
                                        gradient_mask[y, x] = 255
    
    # 可視化結果
    plt.figure(figsize=(20, 5))
    
    # 原始圖片
    plt.subplot(141)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # 梯度大小圖
    plt.subplot(142)
    plt.imshow(gradient_magnitude, cmap='viridis')
    plt.colorbar()
    plt.title('Gradient Magnitude')
    plt.axis('off')
    
    # 方向差異的區域
    plt.subplot(143)
    plt.imshow(gradient_mask, cmap='gray')
    plt.title('Gradient Direction Difference')
    plt.axis('off')
    
    # 疊加顯示
    plt.subplot(144)
    overlay = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB).copy()
    overlay[gradient_mask == 255] = [255, 0, 0]  # 用紅色標記符合條件的區域
    plt.imshow(overlay)
    plt.title('Overlay Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_name}_analysis.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # 保存掩碼
    cv2.imwrite(f"{output_name}_mask.png", gradient_mask)
    
    print(f"分析結果已保存為：{output_name}_analysis.png 和 {output_name}_mask.png")
    
    return gradient_mask

if __name__ == "__main__":
    # 測試圖片處理
    total_start_time = time.time()
    image_path = "./imprint/al_grip/RGB/transform/img3_RGB.png"
    output_name = "gradient_visualization_img3_norm"  # 在这里设置输出文件名
    try:
        gradient_magnitude, gradient_direction, original_img = process_image(image_path)
        visualize_gradients(gradient_magnitude, gradient_direction, original_img, output_name)
        
        # 分析梯度方向
        gx_values = np.cos(gradient_direction) * gradient_magnitude
        gy_values = np.sin(gradient_direction) * gradient_magnitude
        analyze_gradient_directions(gx_values, gy_values, original_img, "gradient_direction_analysis")
        
        total_time = time.time() - total_start_time
        print(f"總運行時間: {total_time:.2f} 秒")
        print(f"梯度可視化已完成，結果已保存為 {output_name}.png 和 {output_name}_magnitude.png")
    except Exception as e:
        print(f"處理圖片時出錯: {str(e)}") 