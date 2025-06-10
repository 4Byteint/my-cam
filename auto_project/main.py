import numpy as np
from picamera2 import Picamera2, Preview
import cv2
import os
import time, board, threading, neopixel

from utils import draw_fps, apply_perspective_transform
from camera_module import Camera
from tflite_segmentation import TFLiteModel
import config
from inference_segmentation import UNetSegmenter

# lock
shared_mask = None
mask_lock = threading.Lock()

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
COLOR_RED = (120, 0, 0)
COLOR_OFF = (0, 0, 0)

COUNT_BLUE = 4
COUNT_GREEN = 6
COUNT_OFF = 4
COUNT_RED = 6

def set_leds_task():
    if (COUNT_BLUE + COUNT_GREEN + COUNT_OFF + COUNT_RED) != LED_COUNT:
        print("Error: count is not equal")
        exit(1)

    pixels.fill(COLOR_OFF)
    for i in range(0, COUNT_BLUE):
        pixels[i] = COLOR_BLUE
    
    for i in range(COUNT_BLUE, COUNT_BLUE + COUNT_GREEN):
        pixels[i] = COLOR_GREEN
    
    for i in range(COUNT_BLUE + COUNT_GREEN, COUNT_BLUE + COUNT_GREEN + COUNT_OFF):
        pixels[i] = COLOR_OFF
        
    for i in range(COUNT_BLUE + COUNT_GREEN + COUNT_OFF, COUNT_BLUE + COUNT_GREEN + COUNT_OFF + COUNT_RED):
        pixels[i] = COLOR_RED

    pixels.show()
        
def show_prediction_result(cam, model, stop_event):
    global shared_mask
    while not stop_event.is_set():
        try:
            frame = cam.get_latest_frame()
            if frame is None:
                continue
            frame = apply_perspective_transform(frame)
            # mask = model.predict(frame)
            all_color, wire_mask, connector_mask = model.predict(frame, return_color=True, save=False)
            # mask_display = cv2.cvtColor(mask * 255)
            mask_display = cv2.cvtColor(all_color, cv2.COLOR_RGB2BGR)
            with mask_lock:
                shared_mask = mask_display      
        except Exception as e:
            print(f"infer thread is error. {e}")
        
def print_all_threads():
    msg = f"現在執行緒數量：{threading.active_count()}\n"
    for t in threading.enumerate():
        msg += f"Name: {t.name}, Deamon:{t.daemon}!\n"
    print(msg)
    
def main():
    cam = Camera(use_undistort=True)
    cam.start()
    set_leds_task()
    stop_event = threading.Event()

    # 初始化模型
    # model = TFLiteModel(config.TFLITE_MODEL_NAME)
    model = UNetSegmenter(config.PTH_MODEL_PATH)
    
    infer_thread = threading.Thread(target=show_prediction_result, args=(cam, model, stop_event), daemon=True)
    infer_thread.start()
    base_count = 0
    try:
        while True:
            frame = cam.get_latest_frame()
            if frame is None:
                continue

            # # 顯示即時攝影機畫面（包含 FPS）
            # fps = cam.get_fps()
            # frame_with_fps = draw_fps(frame, fps)
            # cv2.imshow("Camera View", frame_with_fps)
            cv2.imshow("Camera View", frame)
            with mask_lock:
                if shared_mask is not None:
                    cv2.imshow("Mask", shared_mask)           
            
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('b'):
                base_path = "./dataset/v2"
                img_name = os.path.join(base_path, f"img{base_count}.png")
                frame = apply_perspective_transform(frame)
                cv2.imwrite(img_name, frame)
                base_count += 1
                print(f"已儲存圖片結果：{img_name}")
    finally:
        stop_event.set()
        infer_thread.join()
        cv2.destroyAllWindows()
        cam.close()
        print("main process ends totally.")

if __name__ == "__main__":
    print("start program")
    main()
