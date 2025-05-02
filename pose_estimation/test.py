import cv2, sys
print("Python exe :", sys.executable)
print("OpenCV ver :", cv2.__version__)
print("aruco mod? :", hasattr(cv2, "aruco"))


import cv2
print("OpenCV:", cv2.__version__)
aruco_ok = hasattr(cv2, "aruco")
print("Has aruco?", aruco_ok)
