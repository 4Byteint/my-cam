import time
from picamera2 import Picamera2
import config 
import threading
import numpy as np
import cv2

class Camera:
	def __init__(self, use_undistort=False):
		self.picam2 = Picamera2()
		self.config = self.picam2.create_preview_configuration(main={"size": config.RESOLUTION})
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
		
		self.latest_frame = None
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
		
	def start(self):
		self.running = True
		self.thread = threading.Thread(target=self._update_frame, daemon=True)
		self.thread.start()
	def _grab_frame(self):
		raw_frame = self.picam2.capture_array()
		frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
		if self.use_undistort:
			frame = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
		return frame

	def _update_frame(self):
		while self.running:
			frame = self._grab_frame()
			if frame is None:
				continue
			with self.lock:
				self.latest_frame = frame
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
		

	def get_latest_frame(self):
		""" 供多執行緒安全讀取最新一張影像 """
		with self.lock:
			if self.latest_frame is None:
				return None
			return self.latest_frame.copy()

	def read(self):
		""" 讀取最新影像(不經由快取) """
		return self._grab_frame()
	def close(self):
		self.running = False
		if hasattr(self, 'thread'):
			self.thread.join()
		self.picam2.stop()
		self.picam2.close()