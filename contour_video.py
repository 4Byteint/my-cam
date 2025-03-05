import cv2  # bringing in OpenCV libraries

# 開啟攝像頭
cap = cv2.VideoCapture(0)

while True:
    # 讀取每一幀
    ret, frame = cap.read()
    if not ret:
        break

    # 將圖像轉換為灰度圖像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 計算圖像差異
    blurred_diff = cv2.GaussianBlur(gray, (7, 7), 0)
    ret, binary_im = cv2.threshold(blurred_diff, 25, 255, cv2.THRESH_BINARY)

    # 找到輪廓
    contours, hierarchy = cv2.findContours(binary_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 設定一下drawContours的參數
    contours_to_plot = -1  # 畫全部
    plotting_color = (0, 255, 0)  # 畫綠色框
    thickness = -1

    # 開始畫contours
    with_contours = cv2.drawContours(frame, contours, contours_to_plot, plotting_color, thickness)

    # 標示矩形邊框
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        image = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # 根據最大面積的輪廓畫橢圓並打印中心點和旋轉角度
    if contours:
        required_contour = max(contours, key=cv2.contourArea)
        if len(required_contour) >= 5:  # fitEllipse requires at least 5 points
            ellipse = cv2.fitEllipse(required_contour)
            cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
            
            # 打印橢圓的中心點和旋轉角度
            center, axes, angle = ellipse
            print(f"Ellipse Center: {center}")
            print(f"Ellipse Angle: {angle}")

    # 顯示結果
    cv2.imshow('Frame', frame)

    # 按下 'q' 鍵退出循環
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝像頭並關閉所有窗口
cap.release()
cv2.destroyAllWindows()