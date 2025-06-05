import time
from picamera2 import Picamera2
import config 
import threading
import numpy as np
import cv2

class Camera:
	def __init__(self, use_undistort=False):
		self.picam2 = Picamera2()
		self.config = self.picam2.create_preview_configuration(main={"size":config.RESOLUTION})
		self.picam2.configure(self.config)
		self.picam2.set_controls({
            "AfMode": config.AF_MODE,
            "LensPosition": config.LENS_POSITION,
            "AwbEnable": config.AWB_ENABLE,
            "ColourGains": config.COLOUR_GAINS,
            "ExposureValue": config.EXPOSURE_VALUE
		})
		self.picam2.start()
		time.sleep(1)
		
		self.lastest_frame = None
		self.running = True
		self.lock = threading.Lock()
		
		self.mtx = np.load(config.CAMERA_MATRIX_PATH)
		self.dist = np.load(config.DIST_COEFF_PATH)
		
		self.use_undistort = use_undistort
		if use_undistort:
			h, w = config.RESOLUTION[1], config.RESOLUTION[0]
			newcameramtx, _ = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), alpha=0.9)
			self.map1, self.map2 = cv2.initUndistortRectifyMap(self.mtx, self.dist, None, newcameramtx, (w, h), cv2.CV_16SC2)
	
		
		# for fps 
		self.last_time = time.time()
		self.frame_count = 0
		self.fps = 0.0
		
		self.thread = threading.Thread(target=self._update_frame, daemon=True)
		self.thread.start()
		
	def _update_frame(self):
		while self.running:
			frame = self.picam2.capture_array()
			if self.use_undistort:
				frame = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
			with self.lock:
				self.lastest_frame = frame
				self._update_fps()
				
	def _update_fps(self):
		self.frame_count += 1
		now = time.time()
		elapsed = now - self.last_time
		if elapsed >= 1:
			self.fps = self.frame_count / elapsed
			self.last_time = now
			self.frame_count = 0
	
	def get_fps(self):
		return self.fps
		
	def read(self):
		with self.lock:
			return self.lastest_frame.copy() if self.lastest_frame is not None else None	
			
	def close(self):
		self.running = False
		self.picam2.close()
		
