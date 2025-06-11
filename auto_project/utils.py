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

def draw_fps(frame, fps):
	cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	return frame

def apply_perspective_transform(image):
    H = np.load(config.PERSPECTIVE_MATRIX_PATH)
    warped_image = cv2.warpPerspective(image, H, config.PERSPECTIVE_SIZE)
    return warped_image
