
# Project Folder Structure
1. images/:
Contains input chessboard images (JPEG/PNG) used for calibration.

2. calibration/:
Stores calibration result files:

    camera_matrix.npy: Intrinsic matrix

    dist_coeffs.npy: Distortion coefficients

    rvecs.npy, tvecs.npy: Rotation and translation vectors

3. model/:
train/
inference/

4. model_train/:
put the results of trained model 

5. else/:
stores some trivials code

# Camera Calibration
1. Use a **chessboard pattern**, e.g., 9×6 inner corners.
2. RMS reprojection error: 0.04 pixels (it's good!) then you will get some .npy files in ./calibration/
run
```
python camera_calibration.py
```
# Perspective Correction(transform)

run
```
python 
```


# 
