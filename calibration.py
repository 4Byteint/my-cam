import numpy as np
import cv2 as cv
import glob
chessboard = (4, 4) # corner: rows*cols
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
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (chessboard[0],chessboard[1]), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
cv.destroyAllWindows()
#calibration: 誤差/內參/畸變參數/旋轉向量/平移向量
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# save calibration results
np.save('camera_matrix.npy', mtx)
np.save('dist_coeff.npy', dist)

# 計算新的內參矩陣
h, w = gray.shape[:2]
newcameramtx, _ = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
# 讀取待校正的圖像
img = cv.imread('./calibration/fixed_cam/img2.png')
# 進行去畸變校正
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# 儲存校正後的影像
cv.imwrite('calibresult.png', dst)
