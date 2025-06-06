import numpy as np
from picamera2 import Picamera2, Preview
import cv2
import os
import time, board, threading, neopixel

from utils import draw_fps
from camera_module import Camera
from tflite_segmentation import TFLiteModel
import config

LED_PIN = board.D18
LED_COUNT = 20
LED_BRIGHTNESS = 10
pixels = neopixel.NeoPixel(
    LED_PIN,
    LED_COUNT,
    brightness = LED_BRIGHTNESS,
    auto_write = False
)

COLOR_BLUE = (0, 0, 255)
COLOR_GREEN = (0, 55, 0)
COLOR_RED = (159, 0, 0)
COLOR_OFF = (0, 0, 0)

COUNT_BLUE = 4
COUNT_GREEN = 6
COUNT_OFF = 4
COUNT_RED = 6

def set_leds_task(stop_event):
    
    if (COUNT_BLUE + COUNT_GREEN + COUNT_OFF + COUNT_RED) != LED_COUNT:
        print("Error: count is not equal")
        exit(1)

    pixels.fill(COLOR_OFF)
    
    while True:
        for i in range(0, COUNT_BLUE):
            pixels[i] = COLOR_BLUE
        
        for i in range(COUNT_BLUE, COUNT_BLUE + COUNT_GREEN):
            pixels[i] = COLOR_GREEN
        
        for i in range(COUNT_BLUE + COUNT_GREEN, COUNT_BLUE + COUNT_GREEN + COUNT_OFF):
            pixels[i] = COLOR_OFF
            
        for i in range(COUNT_BLUE + COUNT_GREEN + COUNT_OFF, COUNT_BLUE + COUNT_GREEN + COUNT_OFF + COUNT_RED):
            pixels[i] = COLOR_RED

        pixels.show()
        
        if stop_event.is_set():
            break
            
    pixels.fill((0, 0, 0))
    pixels.show()
    
        
def show_prediction_result(original_frame, mask):
    """分別在兩個視窗中顯示原始圖片和推論結果，保持原始大小"""
    # 將遮罩轉換為彩色圖像以便顯示
    mask_colored = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    
    # 分別顯示在兩個視窗中，保持原始大小
    cv2.imshow("Original Image", original_frame)
    cv2.imshow("Prediction Mask", mask_colored)

def main():
    stop_event = threading.Event()
    led_thread = threading.Thread(target=set_leds_task, args=(stop_event,), daemon=False)
    led_thread.start()
    
    # 初始化模型
    model = TFLiteModel(config.TFLITE_MODEL_PATH)
    
    # picamera
    cam = Camera(use_undistort=True)
    try:
        while True:
        
            frame = cam.read()
            if frame is None:
                continue
                
            # 進行推論
            mask = model.predict(frame)
            
            # 顯示原始圖片和推論結果
            show_prediction_result(frame, mask)
            
            # 顯示即時攝影機畫面（包含 FPS）
            fps = cam.get_fps()
            dst = draw_fps(frame, fps)
            cv2.imshow("Camera View", dst)
            
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('b'):
                base_path = "./imprint/250601"
                base_count = 0
                img_name = os.path.join(base_path, f"img{base_count}.png")
                mask_name = os.path.join(base_path, f"img{base_count}_mask.png")
                cv2.imwrite(img_name, frame)
                cv2.imwrite(mask_name, mask * 255)
                base_count += 1
                print(f"已儲存圖片和推論結果：{img_name}, {mask_name}")
    finally:
        stop_event.set()
        led_thread.join()
        cv2.destroyAllWindows()
        cam.close()
        print("main process ends totally.")

if __name__ == "__main__":
    print("start program")
    main()
