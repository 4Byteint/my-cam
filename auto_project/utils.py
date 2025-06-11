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

def process_folder(base_dir, sample_dir, output_dir):
	"""
	For each image in sample_dir, subtract the corresponding image in base_dir,
	and save the result to output_dir. Assumes matching filenames.
	"""
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	for filename in os.listdir(sample_dir):
		if not filename.lower().endswith('.png'):
			continue
		sample_path = os.path.join(sample_dir, filename)
		base_path = os.path.join(base_dir, filename)
		output_path = os.path.join(output_dir, filename)
		if not os.path.isfile(sample_path) or not os.path.isfile(base_path):
			continue
		sample_img = cv2.imread(sample_path)
		base_img = cv2.imread(base_path)
		if sample_img is None or base_img is None:
			continue
		# Ensure images are the same size
		if sample_img.shape != base_img.shape:
			print(f"Shape mismatch for {filename}, skipping.")
			continue
		diff_img = cv2.subtract(sample_img, base_img)
		cv2.imwrite(output_path, diff_img)