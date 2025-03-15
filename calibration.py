import numpy as np
import cv2 as cv
import glob
chessboard = (4, 4) 
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros(((chessboard[0]*chessboard[1]),3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard[0],0:chessboard[1]].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('./calibration/fixed_cam/*.png')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (chessboard[0],chessboard[1]), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print(f"Detected {len(corners) if ret else 0} corners in {fname}")
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (chessboard[0],chessboard[1]), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
    else:
        print(f"Warning: Failed to detect chessboard corners in {fname}")
cv.destroyAllWindows()
#calibration: 誤差/內參/畸變參數/旋轉向量/平移向量
h, w = gray.shape[:2]
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
print('mtx: ',mtx)
print('dist: ',dist)
# save calibration results
np.save('camera_matrix.npy', mtx)
np.save('dist_coeff.npy', dist)

newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0.9, (w, h))

# 讀取待校正的圖像
img = cv.imread('./calibration/fixed_cam/img1.png')
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
cv.imwrite("calibresult.png", dst)
print("Calibration completed successfully. Output saved as calibresult.png")
