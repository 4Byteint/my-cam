import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
# from onnx2tflite import onnx_converter
from utils import apply_perspective_transform
import config

# 模型轉換相關設定

OUTPUT_PATH = "../model_train/tflite_model"


"""def onnx_to_tflite():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    res = onnx_converter(
        onnx_model_path=ONNX_PATH,
        need_simplify=True,
        output_path=OUTPUT_PATH,
        target_formats=["tflite"],
    )
    print("轉換結果：", res)
    print("onnx 轉換為 tflite 完成")
"""
class TFLiteModel:
    """TFLite 模型推論類別"""
    def __init__(self, model_path):
        """
        初始化 TFLite 模型
        
        Args:
            model_path (str): TFLite 模型檔案路徑
        """
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3]  # (height, width)
        
    def preprocess(self, image):
        raw_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = apply_perspective_transform(raw_image, config.PERSPECTIVE_MATRIX_PATH, config.PERSPECTIVE_SIZE)
        
        # 正規化
        input_data = image.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        return input_data
        
    def predict(self, image):
        """
        執行模型推論
        
        Args:
            image (np.ndarray): 輸入影像 (BGR 格式)
            
        Returns:
            np.ndarray: 分割遮罩 (0-1 範圍)
        """
        # 預處理
        input_data = self.preprocess(image)
        
        # 執行推論
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # 生成遮罩
        if output_data.ndim == 4 and output_data.shape[-1] > 1:
            mask = np.argmax(output_data[0], axis=-1).astype(np.uint8)
        else:
            mask = (output_data[0, ..., 0] > 0.5).astype(np.uint8)
     
        return mask
    
    def save_prediction(self, mask, save_path):
        """
        儲存預測結果
        
        Args:
            mask (np.ndarray): 預測遮罩
            save_path (str): 儲存路徑
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, mask * 255)

def check_precision(onnx_path):
    """檢查 ONNX 模型的精度設定"""
    import onnxruntime as ort

    sess = ort.InferenceSession(ONNX_PATH)

    # 檢查第一個輸入
    inp = sess.get_inputs()[0]
    print("Input name:", inp.name)
    print("Input type:", inp.type)      # e.g. 'tensor(float)' or 'tensor(float16)'

    # 檢查第一個輸出
    out = sess.get_outputs()[0]
    print("Output name:", out.name)
    print("Output type:", out.type)

def main():
    """主程式：示範如何使用 TFLiteModel"""
    # 範例：讀取圖片並進行推論
    model = TFLiteModel(config.TFLITE_MODEL_PATH)
    
    # 讀取測試圖片
    test_image_path = "../imprint/dataset.label/final_dataset_voc/PngImages/img67.png"
    image = cv2.imread(test_image_path)
    if image is None:
        raise FileNotFoundError(f"無法讀取圖片：{test_image_path}")
    
    # 執行推論
    mask = model.predict(image)
    
    # 儲存結果
    save_path = "../tflite_transfer/tflite_predict/tflite_predict_img67.png"
    model.save_prediction(mask, save_path)
    print(f"預測結果已儲存至：{save_path}")

if __name__ == "__main__":
    main()
