import cv2  # bringing in OpenCV libraries

base_path = './trans-processing/inference/2-RG/cropped/img1_baseline.png'
sample_path = './trans-processing/inference/2-RG/cropped/img3.png'

# 讀取圖像
base_image = cv2.imread(base_path)
sample_image = cv2.imread(sample_path)
sample_copy = sample_image.copy()
diff = cv2.absdiff(base_image, sample_image)
sample_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', sample_gray)

blurred_diff = cv2.GaussianBlur(sample_gray, (7, 7), 0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 16))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 16))
dilated = cv2.dilate(blurred_diff, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel2, iterations=1)
ret,binary_im = cv2.threshold(eroded,25, 255,cv2.THRESH_BINARY)
cv2.imshow('binary', binary_im)

contours,hierarchy = cv2.findContours(binary_im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#設定一下drawContours的參數
contours_to_plot= -1 #畫全部
plotting_color= (0,255,0)#畫綠色框
thickness= -1
#開始畫contours
with_contours = cv2.drawContours(sample_copy,contours, contours_to_plot, plotting_color,thickness)
cv2.imshow('contours', with_contours)
#標示矩形邊框
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    image = cv2.rectangle(sample_copy,(x,y),(x+w,y+h),(0,255,255),2)
cv2.imshow('contours', image)

# 根據最大面積的輪廓畫出矩形
required_contour = max(contours, key = cv2.contourArea)
x,y,w,h = cv2.boundingRect(required_contour)
img_copy2 = cv2.rectangle(sample_image, (x,y),(x+w, y+h),(0,255,255),2)
cv2.imshow('largest contour', img_copy2)

# 根據最大面積的輪廓畫橢圓
if len(required_contour) >= 5:  # fitEllipse requires at least 5 points
    ellipse = cv2.fitEllipse(required_contour)
    cv2.ellipse(sample_image, ellipse, (0, 255, 0), 2)
    cv2.imshow('largest contour ellipse', sample_image)
    # 打印橢圓的中心點和旋轉角度
    center, axes, angle = ellipse
    print(f"Ellipse Center: {center}, Ellipse Angle: {angle}")

cv2.waitKey(0)
cv2.destroyAllWindows()

