# unet_segmentation_train.py
# 📦 輕量化 U-Net 語意分割訓練流程（背景 / 線 / 接頭）

from datetime import datetime
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
import config
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import random

# ============ 輕量化 U-Net 模型（3 類別輸出） ============
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2, ceil_mode=True)
        self.enc2 = conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2, ceil_mode=True)

        self.bottleneck = conv_block(32, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = conv_block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = conv_block(32, 16)

        self.final = nn.Conv2d(16, out_channels, 1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))

        up2 = self.up2(bottleneck)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        return self.final(dec1)

# ============ 自訂 Dataset ============
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, train=True, use_augmentation=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.transform = T.ToTensor()
        # self.transform_image, self.transform_mask = self.get_transforms(train, use_augmentation)
        self.train = train
        self.use_augmentation = use_augmentation
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.image_list[idx])
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)
        # 轉成 cv 可用格式
        image = np.array(image).astype(np.uint8)
        mask = np.array(mask).astype(np.uint8)

        if self.train and self.use_augmentation:
            image, mask = self.augment(image, mask)

        # 轉為 tensor 格式
        image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        mask = torch.from_numpy(mask).long()

        return image, mask
    
    
    def augment(self, image, mask):
        h, w = image.shape[:2]

        # ✅ 小角度旋轉 ±15°，仿射變換（image=bilinear, mask=nearest）
        angle = random.uniform(-15, 15)
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        # ✅ 光強變化
        brightness_factor = random.uniform(0.7, 1.3)
        image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)

        # ✅ 模糊
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (5, 5), 0)

        # ✅ 加入高斯雜訊
        if random.random() < 0.5:
            noise = np.random.normal(0, 10, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return image, mask
    
    
    
# ============ 計算準確率 ============
def calculate_accuracy(pred, target):
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).float().sum()
    total = target.numel()
    return (correct / total).item()

# ============ 儲存預測結果圖像 ============
def denormalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """還原 Normalize 後的張量回原始顏色區間"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor.clamp(0, 1)
def save_prediction_image(image, mask, pred, save_path):
    image_np = image.permute(1, 2, 0).cpu().numpy()
    mask_np = mask.cpu().numpy()
    pred_np = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title("Input Image")
    plt.subplot(1, 3, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.title("Ground Truth")
    plt.subplot(1, 3, 3)
    plt.imshow(pred_np, cmap='gray')
    plt.title("Predicted Mask")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ============ 訓練流程 ============
def train_model(model, dataloaders, criterion, optimizer, num_epochs=20, lr=1e-3):
    best_loss = float('inf')
    best_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"./model_train/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch+1}/{num_epochs}")

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_acc = 0.0

            for images, masks in tqdm(dataloaders[phase]):
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                if masks.ndim == 4:
                    masks = masks.squeeze(1)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    acc = calculate_accuracy(outputs, masks)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_acc += acc * images.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_acc / len(dataloaders[phase].dataset)

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            print(f"{phase} Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_epoch = epoch + 1
                best_model_path = f"{save_dir}/unet-epoch{epoch+1}-lr{lr}.pth"
                torch.save(model.state_dict(), best_model_path)
                print(f"✅ 儲存最佳模型：{best_model_path}")

    # 打印最佳结果
    print(f"\n🎯 訓練完成！最佳結果：")
    print(f"   最佳 Epoch: {best_epoch}")
    print(f"   最佳驗證 Loss: {best_loss:.4f}")
    print(f"   最佳驗證 Accuracy: {best_acc:.4f}")

    # 儲存 loss 與 acc 曲線
    curve_path = f"{save_dir}/unet-epoch{num_epochs}-lr{lr}_curve.png"
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path)
    print(f"✅ 已儲存訓練曲線：{curve_path}")

    # 儲存預測圖像
    model.eval()
    sample_img, sample_mask = dataloaders['val'].dataset[0]
    with torch.no_grad():
        pred = model(sample_img.unsqueeze(0).to(device))
        
    # denorm_img = denormalize(sample_img.cpu())
    predict_path = f"{save_dir}/unet-epoch{num_epochs}-lr{lr}_predict.png"
    save_prediction_image(sample_img, sample_mask, pred.cpu(), predict_path)
    print(f"✅ 已儲存範例預測圖：{predict_path}")
    # 印出訓練總時長
    total_seconds = (datetime.now() - datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")).total_seconds()
    print(f"⏱️ 訓練總時長：{total_seconds:.2f} 秒")

# ============ 主程式入口 ============
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 使用設備：", device)

    image_dir = "./dataset/v1/demo_dataset_voc/PngImages"
    mask_dir = "./dataset/v1/demo_dataset_voc/SegmentationClass"
    
    
    # 資料集與 DataLoader
    train_dataset = SegmentationDataset(image_dir, mask_dir, train=True, use_augmentation=True)
    val_dataset = SegmentationDataset(image_dir, mask_dir, train=False, use_augmentation=False)
    
    train_len = int(0.8 * len(train_dataset))
    val_len = len(val_dataset) - train_len
    train_set, val_set = random_split(train_dataset, [train_len, val_len])
    
    dataloaders = {
        'train': DataLoader(train_set, batch_size=4, shuffle=True),
        'val': DataLoader(val_set, batch_size=4)
    }

    model = UNet(in_channels=3, out_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_model(model, dataloaders, criterion, optimizer, num_epochs=300, lr=lr)