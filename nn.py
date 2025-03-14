import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from tqdm import tqdm

# =============== 1. 定義 PyTorch Dataset ===============
class SurfaceGradientDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.inputs = self.data[['x', 'y', 'R', 'G']].values.astype(np.float32)
        self.outputs = self.data[['Gx', 'Gy']].values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.outputs[idx], dtype=torch.float32)
        return x, y

# =============== 2. 創建數據集 ===============
csv_file = "lookup_table.csv"
dataset = SurfaceGradientDataset(csv_file)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# =============== 3. 定義神經網路 ===============
class GradientNN(nn.Module):
    def __init__(self):
        super(GradientNN, self).__init__()
        self.fc1 = nn.Linear(4, 32)
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
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# =============== 4. 訓練模型（累積顯示進度條） ===============
epochs = 100
train_loss_history = []
val_loss_history = []
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=True, dynamic_ncols=True)
    for inputs, targets in train_bar:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        train_bar.set_postfix(loss=f"{loss.item():.6f}")
    train_loss = total_loss / len(train_loader)  # 平均化 Train Loss
    train_loss_history.append(train_loss)

    # 驗證模型
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=True, dynamic_ncols=True)
        for inputs, targets in val_bar:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            val_bar.set_postfix(loss=f"{loss.item():.6f}")
    val_loss = val_loss / len(val_loader)
    val_loss_history.append(val_loss)
    # 讓 tqdm 寫入而不覆蓋
    tqdm.write(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {total_loss:.6f} | Val Loss: {val_loss:.6f}")

# =============== 5. 測試模型 ===============
model.eval()
X_test = torch.tensor([[300, 260,65,167]], dtype=torch.float32)
with torch.no_grad():
    predicted_gradient = model(X_test)
print("Predicted Gradients (Gx, Gy):", predicted_gradient.detach().numpy())

# =============== 6. 儲存模型 ===============
torch.save(model.state_dict(), "gradient_model.pth")
print("模型已儲存：gradient_model.pth")
