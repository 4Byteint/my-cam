from train_segmentation import UNet
import torch
import onnx
import onnxruntime as ort
import numpy as np
import config

def pth_to_onnx():
    # 1. 載入 UNet 模型並載入權重
    model = UNet()
    model.load_state_dict(torch.load(config.PTH_MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()

    # 2. 建立固定測試輸入（符合訓練尺寸）
    test_arr = np.random.randn(1, 3, config.MODEL_INPUT_SIZE[0], config.MODEL_INPUT_SIZE[1]).astype(np.float32)
    dummy_input = torch.from_numpy(test_arr)

    # 3. 得到 PyTorch 推論結果
    with torch.no_grad():
        torch_output = model(dummy_input).numpy()

    # 4. 匯出為 ONNX
    onnx_path = config.ONNX_MODEL_PATH
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes=None,         # 關鍵：強制使用靜態 shape
        verbose=True               # 可選：顯示 ONNX 結構資訊
    )

    # 5. 檢查 ONNX 格式是否正確
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX 結構檢查通過")

    # 6. 使用 ONNX Runtime 推論
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {"input": test_arr}
    ort_output = ort_session.run(None, ort_inputs)[0]

    # 7. 比對輸出差異
    max_diff = np.abs(torch_output - ort_output).max()
    print(f"最大輸出差異值：{max_diff:.6f}")



def onnx_to_tflite(onnx_path, output_path):
    from onnx2tflite import onnx_converter
    res = onnx_converter(
            onnx_model_path = onnx_path,
            need_simplify = True,
            output_path = output_path,
            target_formats = ["tflite"],
        )
    print("轉換結果：", res)
    print("onnx 轉換為 tflite 完成")
    
if __name__ == "__main__":
    pth_to_onnx()
    onnx_to_tflite(config.ONNX_MODEL_PATH, config.TFLITE_MODEL_PATH)

