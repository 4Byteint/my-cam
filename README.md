# Camera Calibration
run
```
python calibration.py
```
1. Use a **chessboard pattern**, e.g., 9×6 inner corners.
2. RMS reprojection error: 0.05 pixels (it's good!)
then you will get 2 .npy files in ./calibration/

# Perspective Correction
1. 
I use Homography to transfer 4 points into a knowed ratio size of rectangle
run
```

```
then, it prints 4 points of rectrangle at the frame(counterwise)
2.
