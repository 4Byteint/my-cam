import cv2

# 讀取圖片
image_path = "C:/Jill/Code/camera/trans-processing/trans-wire.png"

image = cv2.imread(image_path)

# 設定裁剪區域的座標 (x1, y1) 為左上角, (x2, y2) 為右下角
x1, y1 = 210, 0  # 左上角
x2, y2 = 462, 335  # 右下角

# 確保 x1, x2, y1, y2 順序正確
x1, x2 = min(x1, x2), max(x1, x2)
y1, y2 = min(y1, y2), max(y1, y2)

# 裁剪影像
cropped_image = image[y1:y2, x1:x2]

# 顯示裁剪結果
cv2.imshow("Cropped Image", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 存檔
cv2.imwrite("wire-cropped.png", cropped_image)
print("ok!")
