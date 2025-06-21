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
import cv2
import config
from pose_estimation import PoseEstimation

class UNetSegmenter:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        self.model = UNet(in_channels=3, out_channels=3).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.transform = T.Compose([
            T.ToTensor(),
            # T.Normalize(mean=(0, 0, 0), std=(1.0, 1.0, 1.0))
        ])
        self.model.eval()

    def predict(self, frame, original_filename=None, return_color=False, output_dir="./predict/v1", save=False):
        """
        frame: 相機拍攝的 BGR 
        item: 'all'、'wire'、'connector'
        save: 是否儲存分割結果圖
        return_color: 是否輸出彩色標註圖
        output_dir: 圖像儲存路徑
        return: pred_mask（np.ndarray）
        """
        if save:
            self._create_dir(output_dir)
        try:
            if original_filename is not None:
                base_name = os.path.splitext(os.path.basename(original_filename))[0]
            else:
                base_name = time.strftime("%Y%m%d_%H%M%S")
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = self.transform(image).unsqueeze(0).to(self.device) # ?
            start_time = time.time()
            # 推論
            with torch.no_grad():
                output = self.model(image_tensor)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            end_time = time.time()
            print(f"推論完成, 耗時: {end_time - start_time:.2f}秒")
            
            # === 建立二值圖（OpenCV 用） ===
            wire_mask = ((pred_mask == 1).astype(np.uint8)) 
            connector_mask = ((pred_mask == 2).astype(np.uint8)) 
            # --- 彩圖 all（背景黑，wire藍，connector綠） ---
            h, w = pred_mask.shape
            all_color = np.zeros((h, w, 3), dtype=np.uint8)
            all_color[pred_mask == 1] = (0, 0, 255)    # wire → blue
            all_color[pred_mask == 2] = (0, 255, 0)    # connector → green
                
            # 根据选择的类别保存结果
            if save:
                # 存彩色圖
                wire_mask = wire_mask * 255
                connector_mask = connector_mask * 255
                Image.fromarray(all_color).save(os.path.join(output_dir, f"{base_name}_color_all.png"))
                Image.fromarray(wire_mask).save(os.path.join(output_dir, f"{base_name}_color_wire.png"))
                Image.fromarray(connector_mask).save(os.path.join(output_dir, f"{base_name}_color_connector.png"))
            
            return all_color, wire_mask, connector_mask # RGB; binary; binary
            
        
        except Exception as e:
            print(f"處理失敗: {str(e)}")
            return None
            
    def _create_dir(self, path):
        os.makedirs(path, exist_ok=True)
    
if __name__ == "__main__":
    model = UNetSegmenter(config.PTH_MODEL_PATH)
    img_path = "./dataset/experiment/img2.png"
    frame = cv2.imread(img_path)
    all_color, wire_mask, connector_mask = model.predict(frame, original_filename=img_path, save=True, output_dir="./dataset/experiment/predict")
   