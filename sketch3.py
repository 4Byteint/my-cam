import cv2  # bringing in OpenCV libraries

base_path = 'C:/Jill/Code/camera/trans-processing/inference/2-RG/cropped/img0_baseline.png'
sample_path = 'C:/Jill/Code/camera/trans-processing/inference/2-RG/cropped/img0.png'

# 讀取圖像
base_image = cv2.imread(base_path)
sample_image = cv2.imread(sample_path)

# 將圖像轉換為灰度圖像
base_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
sample_gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)

# 計算圖像差異
diff_image = cv2.absdiff(base_gray, sample_gray)

# 偵測邊緣
edges = cv2.Canny(diff_image, 30, 130)

# 顯示結果
cv2.imshow('Difference Image', diff_image)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

