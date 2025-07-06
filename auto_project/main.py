import numpy as np
from picamera2 import Picamera2, Preview
import cv2
import os
import time, board, threading, neopixel

from utils import draw_fps, apply_perspective_transform, image_to_world
from camera_module import Camera
from tflite_segmentation import TFLiteModel
import config
from inference_segmentation import UNetSegmenter
from pose_estimation import PoseEstimation
import socket
from collections import deque

# 狀態管理類
class StatusManager:
    def __init__(self):
        self.status_lock = threading.Lock()
        self.current_status = {
            'pose_estimation': None,  # None, 'success', 'failed'
            'inference': None,        # None, 'success', 'failed'
            'wire_conn_img': None,    # None, 'available', 'missing'
        }
        self.last_printed_status = {}
    
    def update_and_print_status(self, status_type, new_status, success_msg=None, fail_msg=None):
        """更新狀態並在狀態改變時印出訊息"""
        with self.status_lock:
            if self.current_status[status_type] != new_status:
                self.current_status[status_type] = new_status
                
                if new_status == 'success' and success_msg:
                    print(success_msg)
                elif new_status == 'failed' and fail_msg:
                    print(fail_msg)
                elif new_status == 'available' and success_msg:
                    print(success_msg)
                elif new_status == 'missing' and fail_msg:
                    print(fail_msg)
                

status_manager = StatusManager()
############################## global variable ###########################
shared_mask = None
shared_wire_img = None
shared_conn_img = None

##########################################################################
mask_lock = threading.Lock()
############################## led setup #################################
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
##################################################################

H_homo = np.load(config.HOMOGRAPHY_MATRIX_PATH)
##################################################################


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
# socket sender
def create_socket_sender(host='127.0.0.1', port=5005):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    return s



def show_prediction_result(cam, model, stop_event, sender_socket):
    global shared_mask
    global shared_wire_img
    global shared_conn_img
    estimator = None
    pos_diff_buffer = deque(maxlen=5)
    angle_diff_buffer = deque(maxlen=5)
    
    last_pos = None
    last_angle = None
    last_status = None
    
    while not stop_event.is_set():
        try:
            frame = cam.get_latest_frame()
            if frame is None:
                continue
            frame = apply_perspective_transform(frame)
            ################################diff infer#################################################
            # diff_img = cv2.absdiff(frame, base_img)
            # mask_display = cv2.cvtColor(all_color, cv2.COLOR_RGB2BGR)
            # mask = model.predict(frame)
            ###########################################################################################
            all_color, wire_mask, connector_mask = model.predict(frame, return_color=True, save=False)
            mask_display = all_color # RGB
            
            # 更新推理狀態為成功
            status_manager.update_and_print_status('inference', 'success')
            
            conn_img = None
            wire_img = None
            #################################################################################
            ############################ pose estimation success ############################
            #################################################################################
            if wire_mask is not None and connector_mask is not None:
                estimator = PoseEstimation(wire_mask, connector_mask)
                if estimator.is_success():
                    (pos, angle, conn_img, wire_img) = estimator.result
                    
                    ###################### sliding window ############################
                    dx, dy, d_angle = 0, 0, 0
                    is_update = False
                    
                    if last_pos is not None and last_angle is not None:
                        dx = pos[0] - last_pos[0]
                        dy = pos[1] - last_pos[1]
                        d_angle = angle - last_angle

                        pos_diff_buffer.append((dx**2 + dy**2)**0.5)
                        angle_diff_buffer.append(abs(d_angle))
                        if len(pos_diff_buffer) == pos_diff_buffer.maxlen:
                            avg_pos_diff = sum(pos_diff_buffer) / len(pos_diff_buffer)
                            avg_angle_diff = sum(angle_diff_buffer) / len(angle_diff_buffer)

                            if avg_pos_diff > 10 or avg_angle_diff > 3:
                                is_update = True
                        else:
                            is_update = True
                    else:
                        is_update = True # first time, default update
                    ##################################################################
                    if is_update:
                        last_pos = pos
                        last_angle = angle
                        success_msg = f"角度為 {angle:.2f}°，中點為 ({pos[0]}, {pos[1]})"
                        status_manager.update_and_print_status('pose_estimation', 'success', success_msg)
                        world_pos = image_to_world(pos, H_homo)
                        last_status = True
                    
                        # socket send
                        try:
                            message = f"{world_pos[0]:.2f},{world_pos[1]:.2f},{angle:.2f}\n"
                            sender_socket.sendall(message.encode('utf-8'))
                            print(f"send message success. {message}")
                        except Exception as e:
                            print(f"[!] send message failed. {e}")
                            
                #################################################################################
                ############################ pose estimation failed #############################
                #################################################################################
                else:
                    try:
                        if last_status != False:
                            message = "nan,nan,nan\n"
                            sender_socket.sendall(message.encode('utf-8'))
                            last_status = False
                            print(f"send nan message success. {message}")
                    except Exception as e:
                        print(f"[!] send nan message failed. {e}")
                    
                    status_manager.update_and_print_status(
                        'pose_estimation', 
                        'failed', 
                        fail_msg="[!] pose estimation failed."
                    )
                #################################################################################

            # 儲存 wire_mask 和 connector_mask
            with mask_lock:
                shared_mask = mask_display 
                shared_wire_img = wire_img
                shared_conn_img = conn_img
                
        except Exception as e:
            status_manager.update_and_print_status(
                'inference', 
                'failed', 
                fail_msg=f"[!] infer failed. {e}"
            )

def main():
    ##############################################################################################
   
    ##############################################################################################`
    # camera setup 
    ##############################################################################################
    cam = Camera(use_undistort=True)
    cam.start()
    set_leds_task()
    stop_event = threading.Event()
    sender_socket = create_socket_sender()
    # 初始化模型
    # model = TFLiteModel(config.TFLITE_MODEL_NAME)
    model = UNetSegmenter(config.PTH_MODEL_PATH)
    ##############################################################################################
    
    infer_thread = threading.Thread(target=show_prediction_result, args=(cam, model, stop_event, sender_socket), daemon=True)
    # infer_thread = threading.Thread(target=show_prediction_result, args=(cam, model, stop_event), daemon=True)
    infer_thread.start()
    base_count = 1
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
                    predict_mask = cv2.cvtColor(shared_mask, cv2.COLOR_RGB2BGR)  # 轉換為 BGR 格式以便顯示
                    
                    cv2.imshow("Mask", predict_mask)
                    if shared_wire_img is not None and shared_conn_img is not None:
                        cv2.imshow("wire Mask", shared_wire_img)
                        cv2.imshow("conn Mask", shared_conn_img)
                    else:
                        status_manager.update_and_print_status('wire_conn_img', 'missing', fail_msg="[!] 沒有回傳 wire_img 或 conn_img")
                        
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('b'):
                base_path = "./dataset/demo_test_1"
                img_name = os.path.join(base_path, f"img{base_count}.png")
                frame = apply_perspective_transform(frame)
                cv2.imwrite(img_name, frame)
                base_count += 1
                print(f"已儲存圖片結果：{img_name}")
    finally:
        stop_event.set()
        infer_thread.join()
        sender_socket.close()
        cv2.destroyAllWindows()
        cam.close()
        print("main process ends totally.")

if __name__ == "__main__":
    print("start program")
    main()
