import cv2
import numpy as np

def sort_points(points):
    """將四個點按照左上、右上、右下、左下的順序排序"""
    # 計算中心點
    center = np.mean(points, axis=0)
    
    # 將點分為上下兩組
    top_points = []
    bottom_points = []
    
    for point in points:
        if point[1] < center[1]:  # y坐標小於中心點的是上方的點
            top_points.append(point)
        else:
            bottom_points.append(point)
    
    # 排序上方的點（左到右）
    top_points = sorted(top_points, key=lambda x: x[0])
    # 排序下方的點（右到左）
    bottom_points = sorted(bottom_points, key=lambda x: x[0], reverse=True)
    
    # 合併點
    return np.array([top_points[0], top_points[1], bottom_points[0], bottom_points[1]])

def find_trapezoid_points(image_path):
    """找出圖像中的梯形點"""
    # 讀取圖像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 進行二值化處理
    _, thresh = cv2.threshold(gray, 70, 100, cv2.THRESH_BINARY)
    cv2.imshow('Detected Trapezoid', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 找輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 遍歷輪廓，找出接近梯形的形狀
    for cnt in contours:
        # 近似多邊形
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # 如果找到四個點，則可能是梯形
        if len(approx) == 4:
            # 提取點的坐標
            points = np.array([point[0] for point in approx])
            
            # 按照左上、右上、右下、左下的順序排序點
            sorted_points = sort_points(points)
            
            # 畫出梯形和點
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
            for i, point in enumerate(sorted_points):
                x, y = point
                print(f"Point {i + 1}: ({x}, {y})")
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            
            return sorted_points, image
    
    return None

def main():
    # 在這裡修改圖像路徑
    image_path = "./calibration/demo/flipped/img0_calib_step.png"
    
    # 找出梯形點
    points, draw_image = find_trapezoid_points(image_path)
    
    if points is not None:
        print("\n排序後的點（左上、右上、右下、左下）：")
        for i, point in enumerate(points):
            print(f"Point {i + 1}: ({point[0]}, {point[1]})")
        
        # 顯示結果
        cv2.imshow('Detected Trapezoid', draw_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("未找到梯形")

if __name__ == "__main__":
    main()
