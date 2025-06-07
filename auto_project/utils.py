#def find_mask_centorid():
	#if mask=='wire':
		#print('wire coordinate: ')
		#if mask == 'connector':
			#print('conncetor centorid: ')

#def draw():
import cv2
import numpy as np

def draw_fps(frame, fps):
	cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	return frame

def apply_perspective_transform(image, H_path, size):
    H = np.load(H_path)
    warped_image = cv2.warpPerspective(image, H, size)
    return warped_image
