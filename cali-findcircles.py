import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


imgpath=r'F:\socket\bd\1.bmp'
# 加载图像
image = cv2.imread(imgpath)

# 棋盘格的行数和列数
grid_size = (7, 7)  # 假设棋盘格有7行7列

def getCorners(gray):
    # 使用 findCirclesGrid 函数检测棋盘格
    # flags 参数可以是 cv2.CALIB_CB_SYMMETRIC_GRID 或 cv2.CALIB_CB_ASYMMETRIC_GRID
    # 根据棋盘格的对称性选择
    params = cv2.SimpleBlobDetector_Params()
    params.maxArea = 10e4
    params.minArea = 10
    params.minDistBetweenBlobs = 5
    blobDetector = cv2.SimpleBlobDetector_create(params)

    return cv2.findCirclesGrid(gray, grid_size, cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector, None)


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob(r'G:\pj\Socket\code\bd\*.bmp')
idx=1
for fname in images:
 print('read '+fname)
 img = cv2.imread(fname)
 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 # Find the chess board corners
 ret, corners = getCorners(gray)
 #print(corners)
 # If found, add object points, image points (after refining them)
 if ret == True:
    objpoints.append(objp)
    #corners = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners)
    print('save '+fname)
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (7,7), corners, ret)
    #cv2.imwrite('imgs/chessboard_'+str(idx)+'.png', img)
    idx+=1

print(objpoints)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)



mean_error = 0
for i in range(len(objpoints)):
 imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
 error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
 mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

np.save('camera_matrix.npy', mtx)
np.save('dist_coeffs.npy', dist)
np.save('rvecs.npy', rvecs)
np.save('tvecs.npy', tvecs)

print(mtx)
print('---------------')
print(dist)

img = cv2.imread(r'G:\pj\Socket\code\bd\1.bmp')
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png', dst)
