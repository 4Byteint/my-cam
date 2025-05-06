import os

import cv2


ONNX_PATH = "./model_train/2025-04-20_00-53-10/unet-epoch234-lr0.0001.onnx"
OUTPUT_PATH = "./model_train/tflite_model"
tflite_path = "./model_train/tflite_model/unet-epoch234-lr0.tflite"
os.makedirs(OUTPUT_PATH, exist_ok=True)

def onnx_to_tflite():
    from onnx2tflite import onnx_converter
    res = onnx_converter(
            onnx_model_path = ONNX_PATH,
            need_simplify = True,
            output_path = OUTPUT_PATH,
            target_formats = ["tflite"],
        )
    print("轉換結果：", res)

    print("onnx 轉換為 tflite 完成")

def tflite_test():
    import numpy as np
    from tflite_runtime.interpreter import Interpreter

    # 1) 加载 .tflite 文件
    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # 2) 拿到输入／输出 tensor 信息
    
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    image_path = "./imprint/dataset.label/final_dataset_voc/PngImages/img67.png"
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise FileNotFoundError(f"圖片載入失敗，請確認路徑是否正確：{image_path}")
    h, w = input_details[0]['shape'][1:3]
    if img.shape[:2] != (h, w): # 最後可刪
        raise ValueError(f"圖片尺寸不匹配，預期尺寸：{h}x{w}，實際尺寸：{img.shape[:2]}")
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    input_data = img.astype(np.float32) / 255.0 # normalize to [0,1]
    
    input_data = np.expand_dims(input_data, axis=0)

    # 3) 运行推理
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])  # e.g. (1,H,W,C) or (1,H,W,1)

    # 4) 生成 mask
    if output_data.ndim == 4 and output_data.shape[-1] > 1:
        mask = np.argmax(output_data[0], axis=-1).astype(np.uint8) # get a HxW mask, 對每個pixel的三個值[0,1,2]取 argmax 
    else:
        mask = (output_data[0, ..., 0] > 0.5).astype(np.uint8)
    print("unique mask values:", np.unique(mask))
    # 5) 直接存为灰度图（0/255）
    
    os.makedirs("./tflite_transfer/tflite_predict", exist_ok=True)
    cv2.imwrite("./tflite_transfer/tflite_predict/tflite_predict_img67.png", mask * 255)

if __name__ == "__main__":
    tflite_test()
