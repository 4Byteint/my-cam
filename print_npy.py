import numpy as np
import cv2 as cv
# 假設你的檔名叫 data.npy
mtx = np.load('C:/Jill/Code/camera/auto_project/calibration/camera_matrix.npy')
dist = np.load('C:/Jill/Code/camera/auto_project/calibration/dist_coeff.npy')

mtx_now = np.load('C:/Jill/Code/camera/calibration/camera_matrix.npy')
dist_now = np.load('C:/Jill/Code/camera/calibration/dist_coeff.npy')
print(mtx)
print(dist)
print(mtx_now)
print(dist_now)

# undistort
img = cv.imread('./calibration/final/img12_calib.png')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0.9, (w, h))
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

newcameramtx_now, roi_now = cv.getOptimalNewCameraMatrix(mtx_now, dist_now, (w, h), 0.9, (w, h))
dst_now = cv.undistort(img, mtx_now, dist_now, None, newcameramtx_now)

cv.imshow("./calibration/calibresult_cam.png", dst)
cv.imshow("./calibration/calibresult_cam_now.png", dst_now)
cv.waitKey(0)
cv.destroyAllWindows()
