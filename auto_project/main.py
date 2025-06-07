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
    
        
def show_prediction_result(cam, model, stop_event):
    while not stop_event.is_set():
        try:
            frame = cam.read()
            if frame is None:
                continue
            mask = model.predict(frame)
            mask_display = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
            cv2.imshow("mask", mask_display)
            
            if cv2.waitKey(1) == 27:
                stop_event.set()
                break
        
        except Exception as e:
            print("infer thread is error.")
        
        
def main():
    cam = Camera(use_undistort=True)
    
    stop_event = threading.Event()
    led_thread = threading.Thread(target=set_leds_task, args=(stop_event,), daemon=False)
    led_thread.start()
    
    # 初始化模型
    model = TFLiteModel(config.TFLITE_MODEL_PATH)
    
    infer_thread = threading.Thread(target=show_prediction_result, args=(cam, model, stop_event), daemon=True)
    infer_thread.start()
    base_count = 0
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            # 顯示即時攝影機畫面（包含 FPS）
            fps = cam.get_fps()
            dst = draw_fps(frame, fps)
            cv2.imshow("Camera View", dst)
            print(f"🧵 現在執行緒數量：{threading.active_count()}")
            
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('b'):
                base_path = "./imprint/250601"
                img_name = os.path.join(base_path, f"img{base_count}.png")
                cv2.imwrite(img_name, frame)
                base_count += 1
                print(f"已儲存圖片結果：{img_name}")
    finally:
        stop_event.set()
        infer_thread.join()
        led_thread.join()
        cv2.destroyAllWindows()
        cam.close()
        print("main process ends totally.")

if __name__ == "__main__":
    print("start program")
    main()
