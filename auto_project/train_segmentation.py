# unet_segmentation_train.py
# 📦 輕量化 U-Net 語意分割訓練流程（背景 / 線 / 接頭）

from datetime import datetime
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.unet import UNet

# ============ 自訂 Dataset ============
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.transform = transform
        self.mask_resize = T.Resize((160, 128), interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.image_list[idx])

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(T.Resize((160, 128))(image))

        mask = torch.as_tensor(np.array(self.mask_resize(mask)), dtype=torch.long)

        return image, mask

# ============ 計算準確率 ============
def calculate_accuracy(pred, target):
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).float().sum()
    total = target.numel()
    return (correct / total).item()

# ============ 儲存預測結果圖像 ============
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
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"./model_train/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\n🌟 Epoch {epoch+1}/{num_epochs}")

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
                best_model_path = f"{save_dir}/unet-epoch{epoch+1}-lr{lr}.pth"
                torch.save(model.state_dict(), best_model_path)
                print(f"✅ 儲存最佳模型：{best_model_path}")

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

    predict_path = f"{save_dir}/unet-epoch{num_epochs}-lr{lr}_predict.png"
    save_prediction_image(sample_img, sample_mask, pred.cpu(), predict_path)
    print(f"✅ 已儲存範例預測圖：{predict_path}")


# ============ 主程式入口 ============
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 使用設備：", device)

    image_dir = "./imprint/dataset.label/final_dataset_voc/PNGImages"
    mask_dir = "./imprint/dataset.label/final_dataset_voc/SegmentationClass"

    dataset = SegmentationDataset(image_dir, mask_dir)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    dataloaders = {
        'train': DataLoader(train_set, batch_size=4, shuffle=True),
        'val': DataLoader(val_set, batch_size=4)
    }

    model = UNet(in_channels=3, out_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_model(model, dataloaders, criterion, optimizer, num_epochs=300, lr=lr)