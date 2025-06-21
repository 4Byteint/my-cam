#def find_mask_centorid():
	#if mask=='wire':
		#print('wire coordinate: ')
		#if mask == 'connector':
			#print('conncetor centorid: ')

#def draw():
import cv2
import numpy as np
import config
import os
import socket

def draw_fps(frame, fps):
	cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	return frame

def apply_perspective_transform(image):
    H = np.load(config.PERSPECTIVE_MATRIX_PATH)
    warped_image = cv2.warpPerspective(image, H, config.PERSPECTIVE_SIZE)
    return warped_image

# image to world coordinate
def image_to_world(image_point, H_homo):
    """
    將圖像座標轉換為世界座標
    
    Args:
        image_point: 圖像座標點 [x, y]
        H: 單應性矩陣
    
    Returns:
        世界座標點 (x, y)
    """
    point = np.array([[image_point]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_point = cv2.perspectiveTransform(point, H_homo)
    return transformed_point[0][0][0], transformed_point[0][0][1]