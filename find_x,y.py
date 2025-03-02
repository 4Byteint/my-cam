import cv2

# 全域變數用來存儲選取區域的座標
ref_point = []
cropping = False

def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("Image", image)

image = cv2.imread("C:/Jill/Code/camera/trans-processing/trans-light-none.png")
clone = image.copy()
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", click_and_crop)

while True:
    cv2.imshow("Image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):  # 按 "r" 重新選擇
        image = clone.copy()
    
    elif key == ord("c"):  # 按 "c" 確認裁切
        if len(ref_point) == 2:
            x1, y1 = ref_point[0]
            x2, y2 = ref_point[1]
            print("左上角座標: ({}, {})".format(x1, y1), "右下角座標: ({}, {})".format(x2, y2))
            roi = clone[y1:y2, x1:x2]
            #roi = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
            cv2.imshow("Cropped", roi)
            cv2.waitKey(0)

    elif key == ord("q"):  # 按 "q" 退出
        break

cv2.destroyAllWindows()
