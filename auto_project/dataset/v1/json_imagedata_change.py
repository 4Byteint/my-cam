import os
import json
import base64

# 👉 修改為你的資料夾路徑
folder_path = "./diff_img"

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        json_path = os.path.join(folder_path, filename)
        base_name = os.path.splitext(filename)[0]

        # 尋找對應的 .png 圖片（副檔名不會自動加）
        image_path = os.path.join(folder_path, base_name + ".png")
        if not os.path.exists(image_path):
            print(f"❌ 找不到對應圖片: {image_path}")
            continue

        # 圖片轉 base64
        with open(image_path, "rb") as img_f:
            img_bytes = img_f.read()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        # 讀取 json 並更新 imageData（不改 imagePath）
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["imageData"] = img_b64  # ✅ 只更新 imageData，不改 imagePath

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"✅ imageData 已更新：{filename}")
