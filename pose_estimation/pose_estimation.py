import math
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


    
def find_edge_points(img_path, center):
    img = cv2.imread(img_path)
    img_bgr = img.copy()
    # 轉為灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 取得圖像寬高
    height, width = gray.shape
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 取最大輪廓（通常是白色區塊）
    cnt = max(contours, key=cv2.contourArea)
    # 多邊形擬合，取得邊的頂點
    epsilon = 0.1 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True) #回傳頂點座標
    
    # 取得角點座標 (row, column)
    corners = [tuple(pt[0]) for pt in approx]  # 每個 pt 為 [[x, y]]
    # 計算中心點到各角點的距離
    # 繪製輪廓多邊形（綠線）
    cv2.drawContours(img_bgr, [approx], -1, (0,255,0), 2)
    if len(corners) < 2:
        print("角點不足兩個，無法選出最近的兩點。")
        return None
    # 繪製角點（紅圈）並標註編號
    for (x, y) in corners:
        cv2.circle(img_bgr, (x, y), 5, (0,255,0), -1)
        

    # 顯示結果
    cv2.imshow("Corners", img_bgr)
    # 計算到 center 的距離並排序，取最小兩個
    a, b = center
    # 計算並得到 (距離, (x,y)) 的列表
    dists = [((x - a)**2 + (y - b)**2, (x, y)) for (x, y) in corners]
    dists.sort(key=lambda t: t[0])
    closest_pts = [dists[0][1], dists[1][1]]

    # 標示最接近的兩點為藍色實心圓，並畫連線
    (x1, y1), (x2, y2) = closest_pts
    cv2.circle(img_bgr, (x1, y1), 6, (255, 0, 0), -1)
    cv2.circle(img_bgr, (x2, y2), 6, (255, 0, 0), -1)
    cv2.line(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # 計算並標記中點（黃色）
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(img_bgr, (mx, my), 6, (0, 0, 255), -1)
    cv2.putText(img_bgr, f"M", (mx+5, my-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    
     # 計算垂線方向向量：d=(dx,dy)，垂直向量 p = (-dy, dx)
    dx, dy = x2 - x1, y2 - y1
    # 正規化 p，並決定垂線長度 L（可調）
    p = np.array([-dy, dx], dtype=float)
    norm = np.hypot(p[0], p[1])
    if norm != 0:
        p_unit = p / norm
    else:
        p_unit = p
    L = 100  # 垂線在中點兩側各延伸長度
    pt1 = (int(mx + p_unit[0] * L), int(my + p_unit[1] * L))
    pt2 = (int(mx - p_unit[0] * L), int(my - p_unit[1] * L))
    # 繪製垂線（黃色）
    cv2.line(img_bgr, pt1, pt2, (0, 0, 255), 2)

    # 計算垂線與 +x 軸的夾角（以度為單位）
    # angle = arctan2(p_y, p_x)
    angle_rad = math.atan2(p_unit[1], p_unit[0])
    angle_deg = angle_rad * 180.0 / math.pi
    # 調整角度範圍至 [0,180)
    if angle_deg < 0:
        angle_deg += 180

    # 在中點旁標註角度
    text = f"{angle_deg:.1f}°"
    cv2.putText(img_bgr, text, (mx-10, my-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
    
    cv2.imshow("final", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return (mx, my), 
    
# def find_edge_points_from_video(video_path):
def find_position_form_wire(image_path, min_area=10):
    """
    計算二值圖中白色區域的質心，並在圖上標註。
    如果白色區域面積小於 min_area，則回傳 None 並不顯示圖像。

    參數:
    - image_path: 影像檔路徑
    - min_area: 最小面積閾值 (以像素數計)

    回傳:
    - (c_centroid, r_centroid) 或 None
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # check 
    if gray is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    _, img_bin = cv2.threshold(
        gray,
        127,      # 閾值
        1,           # 大於閾值置 1
        cv2.THRESH_BINARY
    )
    area = np.count_nonzero(img_bin == 1)
    if area < min_area:
        print(f"白色區域面積小於 {min_area} 像素，不顯示圖像")
        return None
    # count center
    rows, cols = np.where(img_bin == 1)   # 取得所有 white pixel 的 (row, col) 座標
    r_centroid = rows.mean()
    c_centroid = cols.mean()
    print("質心 (column, row) = ", (c_centroid, r_centroid))
    center = (int(c_centroid), int(r_centroid))
    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.circle(img_bgr, center, radius=1, color=(0, 0, 255), thickness=2)
    # 在旁邊標註座標（value=255）
    cv2.putText(
        img_bgr,
        f"C({center[0]},{center[1]})",
        (center[0]+10, center[1]-10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.4,
        color=(0, 0, 255),
        thickness=1
    )
    cv2.imshow("All edges", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return center
########################################################
def main():
    # 載入新的圖像
    image_path = "C:/Jill/Code/camera/model_train/predict_final/img9_predict_connector.png"
    image_wire_path = "C:/Jill/Code/camera/model_train/predict_final/img9_predict_wire.png"
    center = find_position_form_wire(image_wire_path, min_area=50)
    if center is not None:
        find_edge_points(image_path, center)
    else:
        print("未找到白色區域")
       

if __name__ == "__main__":
    main()
