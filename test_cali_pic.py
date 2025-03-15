import numpy as np
from picamera2 import Picamera2, Preview
import cv2 as cv
import os


# 加載相機內部參數和畸變參數
camera_matrix = np.load("./camera_matrix_real.npy")
dist_coeff = np.load("./dist_coeff_real.npy")


# 讀取待校正的圖像
img = cv.imread('./calibration/fixed_cam/img2.png')
dst = cv.undistort(img, camera_matrix, dist_coeff)
cv.imwrite("calibresult_2.png", dst)
