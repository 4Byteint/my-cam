import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time  # 添加time模塊

# 檢查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 記錄開始時間
start_time = time.time()

# 定義準確度計算函數
def calculate_accuracy(predictions, targets, threshold=0.1):
    # 計算預測值與實際值之間的歐氏距離
    distances = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=1))
    # 如果距離小於閾值，則認為預測正確
    correct = (distances < threshold).float()
    return correct.mean().item() * 100  # 轉換為百分比

# =============== 1. 定義 PyTorch Dataset ===============
class SurfaceGradientDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.inputs = self.data[['x', 'y', 'R', 'G', 'B']].values.astype(np.float32)
        self.outputs = self.data[['Gx', 'Gy']].values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.outputs[idx], dtype=torch.float32)
        return x, y

# =============== 2. 創建數據集 ===============
csv_file = "./imprint/al_RGB_calib/transform/circles/lookup_table.csv"
dataset = SurfaceGradientDataset(csv_file)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# =============== 3. 定義神經網路 ===============
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

model = GradientNN()
model = model.to(device)  # 將模型移到GPU
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# =============== 4. 訓練模型（累積顯示進度條） ===============
epochs = 100
train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False, dynamic_ncols=True)
    for inputs, targets in train_bar:
        # 將數據移到GPU
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # 收集預測值和目標值
        all_predictions.append(outputs)
        all_targets.append(targets)
        
        train_bar.set_postfix(loss=f"{loss.item():.6f}")
    
    # 計算訓練準確度
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    train_acc = calculate_accuracy(all_predictions, all_targets)
    train_acc_history.append(train_acc)
    
    train_loss = total_loss / total_samples
    train_loss_history.append(train_loss)

    # 驗證模型
    model.eval()
    val_loss = 0.0
    total_val_samples = 0
    all_val_predictions = []
    all_val_targets = []
    
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=True, dynamic_ncols=True)
        for inputs, targets in val_bar:
            # 將數據移到GPU
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_size = inputs.size(0)
            val_loss += loss.item() * batch_size
            total_val_samples += batch_size
            
            # 收集驗證預測值和目標值
            all_val_predictions.append(outputs)
            all_val_targets.append(targets)
            
            val_bar.set_postfix(loss=f"{loss.item():.6f}")
    
    # 計算驗證準確度
    all_val_predictions = torch.cat(all_val_predictions)
    all_val_targets = torch.cat(all_val_targets)
    val_acc = calculate_accuracy(all_val_predictions, all_val_targets)
    val_acc_history.append(val_acc)
    
    val_loss = val_loss / total_val_samples
    val_loss_history.append(val_loss)
    tqdm.write(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

# =============== 5. 測試模型 ===============
model.eval()
X_test = torch.tensor([[232, 273, 78, 134, 45]], dtype=torch.float32).to(device)
with torch.no_grad():
    predicted_gradient = model(X_test)
print("Predicted Gradients (Gx, Gy):", predicted_gradient.cpu().detach().numpy())

# =============== 6. 儲存模型 ===============
torch.save(model.state_dict(), "gradient_model_RGB_norm.pth")
print("儲存 gradient_model_RGB_norm.pth")

# =============== 7. 繪製損失曲線 ===============
plt.figure(figsize=(12, 5))

# 繪製損失曲線
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.title('Training and Validation Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 繪製準確度曲線
plt.subplot(1, 2, 2)
plt.plot(train_acc_history, label='Training Accuracy')
plt.plot(val_acc_history, label='Validation Accuracy')
plt.title('Training and Validation Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves_RGB.png')
plt.close()
print("訓練曲線已保存為：training_curves_RGB.png")

# 計算並顯示總訓練時間
end_time = time.time()
total_time = end_time - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)
print(f"\n總訓練時間: {hours}小時 {minutes}分鐘 {seconds}秒")
