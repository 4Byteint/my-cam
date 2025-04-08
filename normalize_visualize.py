import numpy as np
import matplotlib.pyplot as plt
import cv2

def normalize_gradients(Gx, Gy):
    """
    规范化梯度场
    :param Gx: X方向梯度
    :param Gy: Y方向梯度
    :return: 规范化后的Gx和Gy
    """
    # 计算梯度幅值
    magnitude = np.sqrt(Gx**2 + Gy**2)
    
    # 避免除以零
    magnitude[magnitude == 0] = 1
    
    # 规范化
    Gx_norm = Gx / magnitude
    Gy_norm = Gy / magnitude
    
    return Gx_norm, Gy_norm

def visualize_normalized_gradients(Gx, Gy, Gx_norm, Gy_norm, output_name="normalized_gradients"):
    """
    可视化原始和规范化后的梯度场
    :param Gx: 原始X方向梯度
    :param Gy: 原始Y方向梯度
    :param Gx_norm: 规范化后的X方向梯度
    :param Gy_norm: 规范化后的Y方向梯度
    :param output_name: 输出文件名
    """
    # 创建网格
    y, x = np.mgrid[0:Gx.shape[0]:10, 0:Gx.shape[1]:10]
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # 原始Gx热图
    im1 = axes[0, 0].imshow(Gx, cmap='coolwarm')
    axes[0, 0].set_title("Original Gx")
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 原始Gy热图
    im2 = axes[0, 1].imshow(Gy, cmap='coolwarm')
    axes[0, 1].set_title("Original Gy")
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 规范化Gx热图
    im3 = axes[1, 0].imshow(Gx_norm, cmap='coolwarm')
    axes[1, 0].set_title("Normalized Gx")
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 规范化Gy热图
    im4 = axes[1, 1].imshow(Gy_norm, cmap='coolwarm')
    axes[1, 1].set_title("Normalized Gy")
    plt.colorbar(im4, ax=axes[1, 1])
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f"{output_name}.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # 创建向量场可视化
    plt.figure(figsize=(10, 10))
    plt.quiver(x, y, Gx_norm[::10, ::10], Gy_norm[::10, ::10], 
               scale=50, color='red', alpha=0.6)
    plt.title("Normalized Gradient Vector Field")
    plt.savefig(f"{output_name}_vector_field.png", bbox_inches='tight', dpi=300)
    plt.close()

def process_and_visualize_gradients(Gx, Gy, output_name="normalized_gradients"):
    """
    处理并可视化梯度场
    :param Gx: X方向梯度
    :param Gy: Y方向梯度
    :param output_name: 输出文件名
    """
    # 规范化梯度
    Gx_norm, Gy_norm = normalize_gradients(Gx, Gy)
    
    # 打印统计信息
    print("原始梯度统计:")
    print(f"Gx 范围: [{Gx.min():.4f}, {Gx.max():.4f}]")
    print(f"Gy 范围: [{Gy.min():.4f}, {Gy.max():.4f}]")
    print("\n规范化后梯度统计:")
    print(f"Gx_norm 范围: [{Gx_norm.min():.4f}, {Gx_norm.max():.4f}]")
    print(f"Gy_norm 范围: [{Gy_norm.min():.4f}, {Gy_norm.max():.4f}]")
    
    # 可视化结果
    visualize_normalized_gradients(Gx, Gy, Gx_norm, Gy_norm, output_name)
    
    return Gx_norm, Gy_norm

if __name__ == "__main__":
    # 示例用法
    # 这里需要提供Gx和Gy的值
    # Gx, Gy = ... # 从其他地方获取梯度值
    # process_and_visualize_gradients(Gx, Gy)
    pass 