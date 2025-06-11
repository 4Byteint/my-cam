import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from train_segmentation import UNet
from tqdm import tqdm
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2


class UNetSegmenter:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        self.model = UNet(in_channels=3, out_channels=3).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.transform = A.Compose([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])
        self.model.eval()

    def predict(self, frame, return_color=False, output_dir="./predict/v1", save=False):
        """
        frame: 相機拍攝的 RGB NumPy 圖片
        item: 'all'、'wire'、'connector'
        save: 是否儲存分割結果圖
        return_color: 是否輸出彩色標註圖
        output_dir: 圖像儲存路徑
        return: pred_mask（np.ndarray）
        """
        if save:
            self._create_dir(output_dir)
        
        
        
        try:
            base_name = time.strftime("%Y%m%d_%H%M%S")
            
            augmented = self.transform(image=frame)
            image_tensor =  augmented['image'].unsqueeze(0).to(self.device) # 加上 batch 維度
            
            start_time = time.time()
            # 推論
            with torch.no_grad():
                output = self.model(image_tensor)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            end_time = time.time()
            print(f"推論完成, 耗時: {end_time - start_time:.2f}秒")
            
            # === 建立二值圖（OpenCV 用） ===
            wire_mask = ((pred_mask == 1).astype(np.uint8)) * 255
            connector_mask = ((pred_mask == 2).astype(np.uint8)) * 255
            # --- 彩圖 all（背景黑，wire藍，connector綠） ---
            h, w = pred_mask.shape
            all_color = np.zeros((h, w, 3), dtype=np.uint8)
            all_color[pred_mask == 1] = (0, 0, 255)    # wire → blue
            all_color[pred_mask == 2] = (0, 255, 0)    # connector → green
                
            # 根据选择的类别保存结果
            if save:
                # 存彩色圖
                Image.fromarray(all_color).save(os.path.join(output_dir, f"{base_name}_color_all.png"))
                Image.fromarray(wire_mask).save(os.path.join(output_dir, f"{base_name}_color_wire.png"))
                Image.fromarray(connector_mask).save(os.path.join(output_dir, f"{base_name}_color_connector.png"))
            
            return all_color, wire_mask, connector_mask
            
        
        except Exception as e:
            print(f"處理失敗: {str(e)}")
            return None
            
    def _create_dir(self, path):
        os.makedirs(path, exist_ok=True)