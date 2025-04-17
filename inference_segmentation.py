import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from train_segmentation import UNet
from tqdm import tqdm

def create_dir_if_not_exists(path):
    """创建目录（如果不存在）"""
    if not os.path.exists(path):
        os.makedirs(path)

def load_model(model_path):
    """加载训练好的模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = UNet(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def process_image(image_path, transform):
    """处理输入图像"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image

def visualize_and_save(image, pred_mask, save_path, target_class=None):
    """可视化并保存预测结果"""
    plt.figure(figsize=(12, 4))
    
    # 原始图像
    plt.subplot(1, 3, 1)
    image_np = image.convert('RGB')
    plt.imshow(image_np)
    plt.title("输入图像")
    plt.axis('off')
    
    # 预测掩码
    plt.subplot(1, 3, 2)
    if target_class is not None:
        mask = (pred_mask == target_class).astype(float)
        title = f"预测掩码 (类别 {target_class})"
    else:
        mask = pred_mask.astype(float)
        title = "预测掩码 (所有类别)"
    plt.imshow(mask, cmap='gray')
    plt.title(title)
    plt.axis('off')
    
    # 叠加显示
    plt.subplot(1, 3, 3)
    plt.imshow(image_np)  # 先显示原图
    plt.imshow(mask, alpha=0.5, cmap='jet')  # 再叠加掩码
    plt.title("叠加效果")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='使用训练好的U-Net模型进行图像分割')
    parser.add_argument('--image_dir', type=str, required=True, help='输入图像目录')
    parser.add_argument('--item', type=str, choices=['wire', 'connector', 'all'], required=True, 
                      help='分割目标类型 (wire=类别1, connector=类别2, all=所有类别)')
    
    args = parser.parse_args()
    
    # 固定模型路径
    model_path = "./model_train/2025-04-14_01-56-16/unet-epoch166-lr0.0001.pth"  # 请根据实际模型路径修改
    
    # 创建输出目录
    output_dir = "./model_train/predict"
    create_dir_if_not_exists(output_dir)
    print(f"输出目录: {output_dir}")
    
    # 加载模型
    model, device = load_model(model_path)
    print(f"模型已加载: {model_path}")
    
    # 设置图像转换
    transform = T.Compose([
        T.Resize((160, 128)),
        T.ToTensor(),
    ])
    
    # 处理目录中的所有图像
    image_files = [f for f in os.listdir(args.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)
    print(f"找到 {total_images} 张图片待处理")
    
    # 使用tqdm添加进度条
    for idx, image_file in enumerate(tqdm(image_files, desc="处理进度"), 1):
        image_path = os.path.join(args.image_dir, image_file)
        
        try:
            # 处理图像
            image_tensor, original_image = process_image(image_path, transform)
            image_tensor = image_tensor.to(device)
            
            # 推理
            with torch.no_grad():
                output = model(image_tensor)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # 获取模型名称（不包含扩展名）
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            
            # 根据选择的类别保存结果
            if args.item == 'all':
                # 保存所有类别
                save_path = os.path.join(output_dir, f"{model_name}_predict_all_img{idx}.png")
                visualize_and_save(original_image, pred_mask, save_path)
                
                # 分别保存每个类别
                for class_id, class_name in [(1, 'wire'), (2, 'connector')]:
                    save_path = os.path.join(output_dir, f"{model_name}_predict_{class_name}_img{idx}.png")
                    visualize_and_save(original_image, pred_mask, save_path, class_id)
            else:
                # 保存单个类别
                target_class = 1 if args.item == 'wire' else 2
                save_path = os.path.join(output_dir, f"{model_name}_predict_{args.item}_img{idx}.png")
                visualize_and_save(original_image, pred_mask, save_path, target_class)
            
        except Exception as e:
            print(f"\n✗ 处理失败 [{image_file}]: {str(e)}")
            continue

    print("\n✓ 处理完成!")

if __name__ == '__main__':
    main()
