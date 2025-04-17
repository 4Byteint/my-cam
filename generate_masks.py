import os
import json
import base64
import labelme
import numpy as np
from PIL import Image

# 路徑設定
json_dir = './imprint/dataset.label/json_file/'
mask_dir = './imprint/dataset.label/masks/'
image_dir = './imprint/dataset.label/images/'
os.makedirs(mask_dir, exist_ok=True)

for json_file in sorted(os.listdir(json_dir)):
    if not json_file.endswith(".json"):
        continue

    base = os.path.splitext(json_file)[0]
    json_path = os.path.join(json_dir, json_file)
    image_path = os.path.join(image_dir, base + ".png")
    mask_path = os.path.join(mask_dir, base + ".png")

    print(f"📄 處理：{json_file}")

    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 強制使用原圖補正 imageData
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        imageData = base64.b64encode(image_bytes).decode('utf-8')
        image = Image.open(image_path)
        imageWidth, imageHeight = image.size
    else:
        print(f"❌ 找不到圖片：{image_path}")
        continue

    # 重建乾淨的資料
    clean_data = {
        "shapes": raw_data["shapes"],
        "imagePath": raw_data["imagePath"],
        "imageData": imageData,
        "imageHeight": imageHeight,
        "imageWidth": imageWidth
    }

    try:
        label_file = labelme.LabelFile(**clean_data)
        image_arr = labelme.utils.img_b64_to_arr(label_file.imageData)
        label, _ = labelme.utils.shapes_to_label(image_arr.shape, label_file.shapes, label_names=None)

        print(f"✔️ 唯一值：{np.unique(label)}")
        Image.fromarray(label.astype(np.uint8)).save(mask_path)
        print(f"✅ 儲存成功：{mask_path}\n")

    except Exception as e:
        print(f"❌ 發生錯誤：{json_file} → {e}\n")