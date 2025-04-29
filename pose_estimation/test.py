import cv2
import numpy as np

def detect_fast_corners(img_path, threshold=5, nonmax_suppression=True):
    """
    使用 OpenCV 內建 FAST 偵測器找角點，回傳角點座標列表 [(row, column), …]
    """
    # 讀圖並轉灰階
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 建立 FAST 偵測器
    fast = cv2.FastFeatureDetector_create()
    fast.setThreshold(threshold)
    fast.setNonmaxSuppression(nonmax_suppression)

    # 偵測 keypoints
    keypoints = fast.detect(gray, None)

    # 轉成 (row, column) 座標
    pts = [(int(k.pt[1]), int(k.pt[0])) for k in keypoints]

    # 畫出結果並顯示
    out = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0))
    cv2.imshow("FAST Corners", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return pts

if __name__ == "__main__":
    corners = detect_fast_corners("C:/Jill/Code/camera/model_train/predict_final/img47_predict_connector.png", threshold=20)
    print("FAST 偵測到的角點：", corners)
