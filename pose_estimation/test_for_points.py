import cv2
import numpy as np

def detect_circles(background_path, foreground_path):
    """
    檢測兩張圖片之間的差異，並找出圓形
    
    Args:
        background_path (str): 背景圖片路徑
        foreground_path (str): 前景圖片路徑
        
    Returns:
        tuple: (output_image, circles_info)
            - output_image: 標記了圓形的輸出圖像
            - circles_info: 檢測到的圓形信息列表，每個元素為 (x, y, r)
    """
    # 讀取圖片
    background = cv2.imread(background_path)
    foreground = cv2.imread(foreground_path)

    # 檢查是否讀取成功
    if background is None or foreground is None:
        print("圖片讀取失敗，請確認路徑正確！")
        return None, None

    # 背景相減
    diff = cv2.absdiff(foreground, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (13, 13), 1)

    # Hough 變換找圓形
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=10,
        param1=50,
        param2=20,
        minRadius=0,
        maxRadius=0
    )

    # 複製原圖準備繪圖
    output = foreground.copy()
    circles_info = []

    # 如果找到圓形，就畫出來
    if circles is not None:
        # 將圓心轉換為 float32 類型
        circles = circles[0, :].astype(np.float32)
        
        # 準備次像素優化的參數
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        winSize = (11, 11)
        zeroZone = (-1, -1)
        
        # 對每個圓心進行次像素優化
        for circle in circles:
            x, y, r = circle
            # 將圓心轉換為二維數組格式
            center = np.array([[x, y]], dtype=np.float32)
            # 進行次像素優化
            refined_center = cv2.cornerSubPix(blurred, center, winSize, zeroZone, criteria)
            
            # 獲取優化後的坐標
            refined_x, refined_y = refined_center[0]
            
            # 在圖像上繪製圓形和圓心
            cv2.circle(output, (int(x), int(y)), int(r), (0, 255, 0), 2)        # 外框
            cv2.circle(output, (int(x), int(y)), 2, (0, 0, 255), 3)        # 圓心
            cv2.putText(output, f"({refined_x:.3f},{refined_y:.3f})", (int(x), int(y)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # 座標
            
            # 儲存圓形信息
            circles_info.append((refined_x, refined_y, r))
            print(f"(x={refined_x:.6f}, y={refined_y:.6f}, r={r:.6f})")
    
    return output, circles_info

def main():
    # 圖片路徑
    background_path = "../calibration/demo/transform/img1_points.png"
    foreground_path = "../calibration/demo/transform/img2_points.png"
    
    # 檢測圓形
    output_image, circles_info = detect_circles(background_path, foreground_path)
    
    if output_image is not None:
        # 顯示結果
        cv2.imshow("Detected Circles", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 返回檢測到的圓形信息
        return circles_info
    return None

if __name__ == "__main__":
    main()
