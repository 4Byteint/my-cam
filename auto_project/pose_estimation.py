import math
import cv2
import numpy as np
    
    
class PoseEstimation:
    def __init__(self, wire_mask, conn_mask, min_area=10):
        
        self.wire_mask = self._auto_binarize(wire_mask)
        self.conn_mask = self._auto_binarize(conn_mask)
        
        # self.wire_mask = wire_mask
        # self.conn_mask = conn_mask
        self.min_area = min_area
        self.center = None
        self.conn_corners = None
        self.conn_angle_deg = None
        self.conn_angle_rad = None
        self.conn_bgr = None
        self.wire_bgr = None
        self.mx_my = None
        self.result = self._process()
        
    def _auto_binarize(self, img, threshold=127):
        if img is None:
            return None
        if img.max() > 1:
            _, binary = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
            return binary.astype(np.uint8)
        return img


    def _process(self):
        if self.wire_mask is None or self.conn_mask is None:
            print("[!] wire_mask 或 conn_mask 未提供")
            return None
        
        if not self._find_center_from_wire():
            return None
        
        conn_result = self._analyze_connector()
        if conn_result is None:
            return None
        
        self.mx_my, self.conn_angle_deg, self.conn_bgr = conn_result
        self.wire_bgr = self._draw_results()
        return self.mx_my, self.conn_angle_deg, self.conn_bgr, self.wire_bgr
    
    def is_success(self):
        return self.result is not None
    
    def _find_center_from_wire(self):
        area = np.count_nonzero(self.wire_mask == 1)
        if area < self.min_area:
            print(f"[!] wire 白色區域面積小於 {self.min_area} 像素，不顯示圖像")
            return False
        rows, cols = np.where(self.wire_mask == 1)   # 取得所有 white pixel 的 (row, col) 座標
        r_centroid = rows.mean()
        c_centroid = cols.mean()
        self.center = (int(c_centroid), int(r_centroid))
        return True
    
    def _analyze_connector(self, min_conn_area=1000):
        if self.center is None:
            print("[!] 未找到 wire 的中心點")
            return None
        # 找輪廓
        contours, _ = cv2.findContours(self.conn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        print(f"connecter area: {area}")
        if area < min_conn_area:
            print(f"[!] connector 區域面積小於 {min_conn_area} 像素，無法分析")
            return None        
        
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True) # 回傳頂點座標
        # 取得角點座標 (row, column)
        corners = [tuple(pt[0]) for pt in approx]  # 每個 pt 為 [[x, y]]
        # 計算center到每個角點的距離，並選出最近的兩點
        if len(corners) < 2:
            print("[!] 角點不足兩個，無法選出最近的兩點。")
            return False
        # 計算到 center 的距離並排序，取最小兩個
        a, b = self.center
        # 計算並得到 (距離, (x,y)) 的列表
        dists = [((x - a)**2 + (y - b)**2, (x, y)) for (x, y) in corners]
        dists.sort(key=lambda t: t[0])
        closest_pts = [dists[0][1], dists[1][1]]

        # 標示最接近的兩點為藍色實心圓，並畫連線
        (x1, y1), (x2, y2) = closest_pts
        # 計算並標記中點（黃色）
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
    
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
        
        # 計算垂線與 +x 軸的夾角（以度為單位）
        # angle = arctan2(p_y, p_x)
        self.conn_angle_rad = math.atan2(p_unit[1], p_unit[0])
        self.conn_angle_deg = self.conn_angle_rad * 180.0 / math.pi
        # 調整角度範圍至 [0,180)
        if self.conn_angle_deg < 0:
            self.conn_angle_deg += 180

        # 畫圖
        ## 繪製輪廓多邊形（綠線）
        conn_bgr = cv2.cvtColor(self.conn_mask * 255, cv2.COLOR_GRAY2BGR) 
        cv2.drawContours(conn_bgr, [approx], -1, (0,255,0), 2)
        ## 繪製紅點角點並標註編號
        for (x, y) in corners:
            cv2.circle(conn_bgr, (x, y), 5, (0,255,0), -1)
        ## 繪製最接近的兩點為藍點，並畫連線
        cv2.circle(conn_bgr, (x1, y1), 6, (255, 0, 0), -1)
        cv2.circle(conn_bgr, (x2, y2), 6, (255, 0, 0), -1)
        cv2.line(conn_bgr, (x1, y1), (x2, y2), (255, 0, 0), 1)
        ## 繪製黃色點為中點
        cv2.circle(conn_bgr, (mx, my), 6, (0, 0, 255), -1)
        cv2.putText(conn_bgr, f"M", (mx+5, my-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        ## 繪製與該直線的黃色垂線
        cv2.line(conn_bgr, pt1, pt2, (0, 0, 255), 2)
        ## 在中點旁標註角度
        text = f"{self.conn_angle_deg:.1f}"
        cv2.putText(conn_bgr, text, (mx-10, my-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
        
        return (mx, my), self.conn_angle_deg, conn_bgr # 0/255
                   
    def _draw_results(self):
        wire_bgr = cv2.cvtColor(self.wire_mask * 255, cv2.COLOR_GRAY2BGR)
        if self.center:
            cv2.circle(wire_bgr, self.center, radius=1, color=(0, 0, 255), thickness=2)
            # 在旁邊標註座標（value=255）
            cv2.putText(
                wire_bgr,
                f"C({self.center[0]},{self.center[1]})",
                (self.center[0]+10, self.center[1]-10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                color=(0, 0, 255),
                thickness=1
            )
        return wire_bgr # 0/255
        

########################################################
def main():
    # 載入新的圖像
    image_path = "../model_train/predict_final/img67_predict_connector.png"
    image_wire_path = "../model_train/predict_final/img67_predict_wire.png"
    wire_mask = cv2.imread(image_wire_path, cv2.IMREAD_GRAYSCALE)
    conn_mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)    
    
    estimator = PoseEstimation(wire_mask, conn_mask)

    if estimator.is_success():
        (mx, my), angle, conn_img, wire_img = estimator.result
        print(f"角度為 {angle:.2f}°，中點為 ({mx}, {my})")
        cv2.imshow("Connector Result", conn_img)
        cv2.imshow("Wire Result", wire_img)
    else:
        print("[!] 分析失敗，請檢查輸入圖像或格式")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
