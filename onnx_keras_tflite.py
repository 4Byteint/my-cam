import os

import cv2


ONNX_PATH = "C:/Jill/Code/camera/model_train/2025-04-20_00-53-10/unet-epoch234-lr0.0001.onnx"
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

    # 3) 准备输入数据（示例：单张 224×224 RGB 图）
    img = cv2.imread("C:/Jill/Code/camera/imprint/dataset.label/final_dataset_voc/PngImages/img14.png")
    h, w = input_details[0]['shape'][1:3]
    img_resized = cv2.resize(img, (w, h))
    input_data = np.expand_dims(img_resized.astype(np.float32), axis=0)

    # 3) 运行推理
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])  # e.g. (1,H,W,C) or (1,H,W,1)

    # 4) 生成 mask
    if output_data.ndim == 4 and output_data.shape[-1] > 1:
        mask = np.argmax(output_data[0], axis=-1).astype(np.uint8)
    else:
        mask = (output_data[0, ..., 0] > 0.5).astype(np.uint8)

    # 5) 直接存为灰度图（0/255）
    cv2.imwrite("C:/Jill/Code/camera/tflite_transfer/tflite_predict/tflite_predict_img14.png", mask * 255)

if __name__ == "__main__":
    tflite_test()
