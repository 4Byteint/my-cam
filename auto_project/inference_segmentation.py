import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from models.unet import UNet
from tqdm import tqdm
import time

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
    """保存预测掩码（纯图片）"""
    # 保存预测掩码
    if target_class is not None:
        mask = (pred_mask == target_class).astype(float)
    else:
        mask = pred_mask.astype(float)
    
    plt.figure(figsize=(6, 4))
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_single_image(image_path, model, device, transform, output_dir, item):
    """处理单张图片并返回处理时间"""
    start_time = time.time()
    
    try:
        # 处理图像
        image_tensor, original_image = process_image(image_path, transform)
        image_tensor = image_tensor.to(device)
        
        # 推理
        with torch.no_grad():
            output = model(image_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        # 获取原始图片的文件名（不包含扩展名）
        original_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # 根据选择的类别保存结果
        if item == 'all':
            # 保存所有类别
            save_path = os.path.join(output_dir, f"{original_filename}_predict_all.png")
            visualize_and_save(original_image, pred_mask, save_path)
            
            # 分别保存每个类别
            for class_id, class_name in [(1, 'wire'), (2, 'connector')]:
                save_path = os.path.join(output_dir, f"{original_filename}_predict_{class_name}.png")
                visualize_and_save(original_image, pred_mask, save_path, class_id)
        else:
            # 保存单个类别
            target_class = 1 if item == 'wire' else 2
            save_path = os.path.join(output_dir, f"{original_filename}_predict_{item}.png")
            visualize_and_save(original_image, pred_mask, save_path, target_class)
            
        end_time = time.time()
        processing_time = end_time - start_time
        return True, processing_time
        
    except Exception as e:
        print(f"\n处理失败 [{os.path.basename(image_path)}]: {str(e)}")
        return False, 0

def process_directory(image_dir, model, device, transform, output_dir, item):
    """处理整个目录的图片"""
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)
    print(f"找到 {total_images} 张图片需要处理")
    
    total_time = 0
    successful_images = 0
    
    for image_file in tqdm(image_files, desc="处理进度"):
        image_path = os.path.join(image_dir, image_file)
        success, processing_time = process_single_image(image_path, model, device, transform, output_dir, item)
        
        if success:
            successful_images += 1
            total_time += processing_time
    
    if successful_images > 0:
        avg_time = total_time / successful_images
        print(f"\n处理完成！成功处理 {successful_images} 张图片")
        print(f"平均每张图片处理时间: {avg_time:.4f} 秒")
    else:
        print("\n没有成功处理任何图片")

def inference_segmentation(image_path, item):
    image_path = "test_img.jpg"
    item = "all"  # 可選：'wire'、'connector'、'all'

    model_path = "./model_train/2025-04-20_00-53-10/unet-epoch234-lr0.0001.pth"
    output_dir = "./model_train/predict_final"
    create_dir_if_not_exists(output_dir)

    model, device = load_model(model_path)
    transform = T.Compose([
        T.Resize((160, 128)),
        T.ToTensor(),
    ])
    
    success, processing_time = process_single_image(image_path, model, device, transform, output_dir, item)
    if success:
        print(f"單張圖片處理完成，用時: {processing_time:.4f} 秒")
        
if __name__ == '__main__':
    main()
