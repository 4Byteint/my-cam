import numpy as np
import cv2 as cv
import glob
chessboard = (6, 9)  # 
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros(((chessboard[0]*chessboard[1]),3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard[0],0:chessboard[1]].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('./webcam_calib/*.jpg')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (chessboard[0],chessboard[1]), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print(f"Detected {len(corners) if ret else 0} corners in {fname}")
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (chessboard[0],chessboard[1]), corners2, ret)
    else:
        print(f"Warning: Failed to detect chessboard corners in {fname}")
        
# calibration: 誤差/內參/畸變參數/旋轉向量/平移向量
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print('mtx: ',mtx)
print('dist: ',dist)
np.save('./webcam_calib/camera_matrix.npy', mtx)
np.save('./webcam_calib/dist_coeff.npy', dist)
np.save('./webcam_calib/rvecs.npy', rvecs)
np.save('./webcam_calib/tvecs.npy', tvecs)
# undistort
img = cv.imread('./webcam_calib/WIN_20250430_12_53_58_Pro.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0.9, (w, h))
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
cv.imwrite("./webcam_calib/calibresult_cam.png", dst)
print("Calibration completed successfully. Output saved as ./calibration/calibresult_cam.png")

# reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error
print(f"total Error: {mean_error / len(objpoints)}")

