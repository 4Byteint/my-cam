import math
import cv2
import numpy as np
import config
from utils import image_to_world


    
class PoseEstimation:
    def __init__(self, wire_mask, conn_mask, min_wire_area=500, min_conn_area=1000):
        
        self.wire_mask = self._auto_binarize(wire_mask)
        self.conn_mask = self._auto_binarize(conn_mask)
        self.min_wire_area = min_wire_area
        self.min_conn_area = min_conn_area
        self.center = None
        self.conn_corners = None
        self.conn_angle_deg = None
        self.conn_angle_rad = None
        self.conn_bgr = None
        self.wire_bgr = None
        self.mx_my = None
        self.major_axis = None
        self.minor_axis = None
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
        ###################################################################################
        ###################################有線############################################
        ###################################################################################
        if self._find_center_from_wire():
            conn_result = self._analyze_connector()
            if conn_result is None:
                return None
            else:
                self.mx_my, self.conn_angle_deg, self.conn_bgr = conn_result
                self.wire_bgr = self._draw_results()
                return self.mx_my, self.conn_angle_deg, self.conn_bgr, self.wire_bgr
        else:
            return None
        ###################################################################################
        ############################不檢查是否有線##########################################
        ###################################################################################
        # conn_result = self.only_conn_pose()
        # if conn_result is None:
        #     return None
        # else:
        #     self.mx_my, self.conn_angle_deg, self.conn_bgr = conn_result
        #     return self.mx_my, self.conn_angle_deg, self.conn_bgr
        ###################################################################################
 
    def is_success(self):
        return self.result is not None
    
    def _find_center_from_wire(self):
        area = np.count_nonzero(self.wire_mask == 1)
        print(f"wire area: {area}")
        if area < self.min_wire_area:
            print(f"[!] wire 白色區域面積小於 {self.min_wire_area} 像素，不顯示圖像")
            return False
        rows, cols = np.where(self.wire_mask == 1)   # 取得所有 white pixel 的 (row, col) 座標
        r_centroid = rows.mean()
        c_centroid = cols.mean()
        self.center = (int(c_centroid), int(r_centroid))
        
        # ************** 使用 PCA 找中心點 **************
        points = np.column_stack((cols, rows))  # cols是X，rows是Y
        pca_center = (c_centroid, r_centroid)
        pca_centered = points - pca_center

        # 3. 協方差矩陣
        cov = np.cov(pca_centered, rowvar=False)

        # 4. 特徵分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov)  # 適用對稱矩陣

        # 5. 依特徵值排序（大→小）
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # 6. 主要與次要方向
        # eigenvectors 是2*2矩陣
        self.major_axis = eigenvectors[:, 0]  # 主方向 第一個col 
        self.minor_axis = eigenvectors[:, 1]  # 次方向 第二個col
        
        return True
    
    
    def _analyze_connector(self):
        if self.center is None:
            print("[!] 未找到 wire 的中心點")
            return None
        # 找輪廓
        contours, _ = cv2.findContours(self.conn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        print(f"connecter area: {area}")
        if area < self.min_conn_area:
            print(f"[!] connector 區域面積小於 {self.min_conn_area} 像素，無法分析")
            return None        
        
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True) # 回傳頂點座標
        # 取得角點座標 (row, column)
        corners = [tuple(pt[0]) for pt in approx]  # 每個 pt 為 [[x, y]]
        
        def _find_edge_with_pca(corners):
            edges = []
            num_points = len(corners)
            for i in range(num_points):
                pt1 = np.array(corners[i])
                pt2 = np.array(corners[(i + 1) % num_points])
                # 計算方向向量
                vec = pt2 - pt1
                norm = np.linalg.norm(vec)
                if norm != 0:
                    vec_unit = vec / norm
                else:
                    vec_unit = vec
                # 計算夾角 (絕對值，忽略方向)
                cos_theta = np.clip(vec_unit @ self.major_axis, -1.0, 1.0)
                angle = math.acos(abs(cos_theta))  # 夾角

                edges.append({
                    'pt1': pt1,
                    'pt2': pt2,
                    'angle': angle,
                    'vector': vec_unit
                })

            # 依照夾角從大到小排序（與主方向越垂直越前面）
            edges_sorted = sorted(edges, key=lambda x: x['angle'], reverse=True)
            # 用中位數分組
            threshold = (edges_sorted[len(edges_sorted) // 2 - 1]['angle'] + edges_sorted[len(edges_sorted) // 2]['angle']) / 2
            group_large = [e for e in edges_sorted if e['angle'] >= threshold]
            group_small = [e for e in edges_sorted if e['angle'] < threshold]

            points_large = [e['pt1'] for e in group_large] + [e['pt2'] for e in group_large]
            points_large = np.array(points_large)
            center = np.array(self.center, dtype=float) # tuple to array
            
            if len(points_large) < 2:
                raise ValueError("points_large 中的點數不足2個")
            
            dists = np.linalg.norm(points_large - center, axis=1)
            sorted_idx = np.argsort(dists)
            closest_point1 = points_large[sorted_idx[0]]
            closest_point2 = points_large[sorted_idx[1]]
            closest_point3 = points_large[sorted_idx[2]]

            return closest_point1, closest_point2, closest_point3
        def _exclude_middle_point(pts):
            """
            排除中心點，回傳左右兩邊點
            """
            pts3 = np.asarray(pts, dtype=float)
            dist_mat = np.linalg.norm(pts3[:, None] - pts3[None, :], axis=-1) # 求兩兩距離矩陣
            # 最大距離所對應的兩個 index ⇒ 端點
            i, j = np.unravel_index(np.argmax(dist_mat), dist_mat.shape) # 找出最大距離的索引
            end_idx = {i, j}
            mid_idx = list({0, 1, 2} - end_idx)[0]

            end_points = pts3[[i, j]]
            # middle_point = pts3[mid_idx]

            return end_points
        
        
        closest_pts = _find_edge_with_pca(corners)
        if closest_pts is None:
            print("[!] 無法找到最接近的兩個點")
            return None
        two_ends = _exclude_middle_point(closest_pts)
        
        
        # ****************************************************************** #
        # ********************** 僅使用接近center的兩點 ********************** #
        # ****************************************************************** #
        # # 計算center到每個角點的距離，並選出最近的兩點
        # if len(corners) < 2:
        #     print("[!] 角點不足兩個，無法選出最近的兩點。")
        #     return False
        # # 計算到 center 的距離並排序，取最小兩個
        # a, b = self.center
        # # 計算並得到 (距離, (x,y)) 的列表
        # dists = [((x - a)**2 + (y - b)**2, (x, y)) for (x, y) in corners]
        # dists.sort(key=lambda t: t[0])
        # closest_pts = [dists[0][1], dists[1][1]]
        
        # ****************************************************************** #
        # ************************ 計算角度 ********************************* #
        # ****************************************************************** #
        (x1, y1), (x2, y2) = sorted(two_ends, key=lambda pt: pt[0])  # 依 x 從小到大排序
        # 計算並標記中點（黃色）
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        
        
        
        
        #############################檢查中點的y座標是否比center的y座標大############################
        # if my > self.center[1]:
        #     print(f"[!] 中點y座標({my})比center的y座標({self.center[1]})大，返回None")
        #     return None
        ##########################################################################################
        
        if self.center[1] < my:
            (x1, y1), (x2, y2) = sorted(two_ends, key=lambda pt: pt[0], reverse=True)  # 依 x 從小到大排序
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
             # 計算連線方向向量：d=(dx,dy)
            dx, dy = x2 - x1, y2 - y1
            # 計算垂線方向向量：p = (dy, -dx) 考慮cv坐標系
            p = np.array([dy, -dx], dtype=float)
            # 正規化
            norm = np.hypot(p[0], p[1])
            if norm != 0:
                p_unit = p / norm
            else:
                p_unit = p
            
            # 計算與 +y 軸的夾角（y 軸為向上，即 (0, -1)）
            angle_rad = math.atan2(p_unit[0], -p_unit[1])  # 注意是 (x, -y)
            angle_deg = - math.degrees(angle_rad) # angle轉成正，再反轉方向，回到數學意義上的逆時針(+)
            
            if angle_deg > 0:
                angle_deg = 180 + angle_deg 
            elif angle_deg < 0:
                angle_deg = -180 + angle_deg
        
        else:
        # 計算連線方向向量：d=(dx,dy)
                dx, dy = x2 - x1, y2 - y1
                # 計算垂線方向向量：p = (dy, -dx) 考慮cv坐標系
                p = np.array([dy, -dx], dtype=float)
                # 正規化
                norm = np.hypot(p[0], p[1])
                if norm != 0:
                    p_unit = p / norm
                else:
                    p_unit = p
                
                # 計算與 +y 軸的夾角（y 軸為向上，即 (0, -1)）
                angle_rad = math.atan2(p_unit[0], -p_unit[1])  # 注意是 (x, -y)
                angle_deg = -math.degrees(angle_rad)
        
        if angle_deg == -180 or angle_deg == 180:
                angle_deg = 0
                
        # 儲存角度
        self.conn_angle_rad = angle_rad
        self.conn_angle_deg = angle_deg
        
        # 畫圖
        ## 繪製輪廓多邊形（綠線）
        conn_bgr = cv2.cvtColor(self.conn_mask * 255, cv2.COLOR_GRAY2BGR) 
        cv2.drawContours(conn_bgr, [approx], -1, (0,255,0), 2)
        ## 繪製紅點角點並標註編號
        for (x, y) in corners:
            cv2.circle(conn_bgr, (int(x), int(y)), 5, (0,255,0), -1)
        ## 繪製最接近的兩點為藍點，並畫連線
        cv2.circle(conn_bgr, (int(x1), int(y1)), 6, (255, 0, 0), -1)
        cv2.circle(conn_bgr, (int(x2), int(y2)), 6, (255, 0, 0), -1)
        cv2.line(conn_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        ## 繪製黃色點為中點
        cv2.circle(conn_bgr, (int(mx), int(my)), 6, (0, 0, 255), -1)
        cv2.putText(conn_bgr, f"M", (int(mx+5), int(my-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        ## 繪製與center相反方向的垂線（長度為10）
        L = 10
        pt1 = (int(mx + p_unit[0] * L), int(my + p_unit[1] * L))
        pt2 = (int(mx - p_unit[0] * L), int(my - p_unit[1] * L))
        cv2.line(conn_bgr, pt1, pt2, (0, 0, 255), 2)
        ## 在中點旁標註角度
        text = f"{self.conn_angle_deg:.1f}"
        cv2.putText(conn_bgr, text, (int(mx-10), int(my-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
        
        return (mx, my), self.conn_angle_deg, conn_bgr # 0/255
              
    def only_conn_pose(self):
        """
        只分析 connector 的 pose，不檢查是否有線，並且把找點邏輯改成找離中心點最近的兩個點
        """
        contours, _ = cv2.findContours(self.conn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        print(f"connecter area: {area}")
        if area < self.min_conn_area:
            print(f"[!] connector 區域面積小於 {self.min_conn_area} 像素，無法分析")
            return None        
       
        epsilon = 0.01* cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True) # 回傳頂點座標
        # 取得角點座標 (row, column)
        corners = [tuple(pt[0]) for pt in approx]  # 每個 pt 為 [[x, y]]
        # 計算center到每個角點的距離，並選出最近的兩點
        if len(corners) < 2:
            print("[!] 角點不足兩個，無法選出最近的兩點。")
            return False
        
        ################################# 設定 a,b 座標 ########################################
        h, w = self.conn_mask.shape[:2]
        a = w
        b = h//2
        ##########################################################################################
        # 計算並得到 (距離, (x,y)) 的列表
        dists = [((x - a)**2 + (y - b)**2, (x, y)) for (x, y) in corners]
        dists.sort(key=lambda t: t[0])
        closest_pts = [dists[0][1], dists[1][1]] 
        (x1, y1), (x2, y2) = sorted(closest_pts, key=lambda pt: pt[0])  # 依 x 從小到大排序
        # print(f"x1, y1: {x1}, {y1}, x2, y2: {x2}, {y2}")
        # 標示最接近的兩點為藍色實心圓，並畫連線
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        
        #############################檢查中點的y座標是否比center的y座標大############################
        # if my > self.center[1]:
        #     print(f"[!] 中點y座標({my})比center的y座標({self.center[1]})大，返回None")
        #     return None
        ##########################################################################################
        
 
        if b < my:
            (x1, y1), (x2, y2) = sorted(closest_pts, key=lambda pt: pt[0], reverse=True)  # 依 x 從小到大排序
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
             # 計算連線方向向量：d=(dx,dy)
            dx, dy = x2 - x1, y2 - y1
            # 計算垂線方向向量：p = (dy, -dx) 考慮cv坐標系
            p = np.array([dy, -dx], dtype=float)
            # 正規化
            norm = np.hypot(p[0], p[1])
            if norm != 0:
                p_unit = p / norm
            else:
                p_unit = p
            
            # 計算與 +y 軸的夾角（y 軸為向上，即 (0, -1)）
            angle_rad = math.atan2(p_unit[0], -p_unit[1])  # 注意是 (x, -y)
            angle_deg = math.degrees(angle_rad)
            
            if angle_deg > 0:
                angle_deg = -180 + angle_deg
            elif angle_deg < 0:
                angle_deg = 180 + angle_deg
        
        
        # 計算連線方向向量：d=(dx,dy)
        dx, dy = x2 - x1, y2 - y1
        # 計算垂線方向向量：p = (dy, -dx) 考慮cv坐標系
        p = np.array([dy, -dx], dtype=float)
        # 正規化
        norm = np.hypot(p[0], p[1])
        if norm != 0:
            p_unit = p / norm
        else:
            p_unit = p
        
        # 計算與 +y 軸的夾角（y 軸為向上，即 (0, -1)）
        angle_rad = math.atan2(p_unit[0], -p_unit[1])  # 注意是 (x, -y)
        angle_deg = math.degrees(angle_rad)
        
        # 儲存角度（你也可以改成回傳）
        self.conn_angle_rad = angle_rad
        self.conn_angle_deg = angle_deg
            
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
        ## 繪製與center相反方向的垂線（長度為10）
        L = 10
        pt1 = (int(mx + p_unit[0] * L), int(my + p_unit[1] * L))
        pt2 = (int(mx - p_unit[0] * L), int(my - p_unit[1] * L))
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
            
            pt1 = self.center
            pt2 = (
                int(self.center[0] + self.major_axis[0] * 100),
                int(self.center[1] + self.major_axis[1] * 100)
            )
            cv2.line(wire_bgr, pt1, pt2, (0, 0, 255), 2) # red

            # 次方向線
            pt3 = (
                int(self.center[0] + self.minor_axis[0] * 100),
                int(self.center[1] + self.minor_axis[1] * 100)
            )
            cv2.line(wire_bgr, pt1, pt3, (0, 255, 0), 2) # green

        return wire_bgr # 0/255
        
########################################################
def main():
    # 載入新的圖像
    image_path = "./dataset/experiment/predict/img7_color_connector.png"
    image_wire_path = "./dataset/experiment/predict/img3_color_wire.png"
    H_homo = np.load(config.HOMOGRAPHY_MATRIX_PATH)
    wire_mask = cv2.imread(image_wire_path, cv2.IMREAD_GRAYSCALE)
    conn_mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)    
    
    estimator = PoseEstimation(wire_mask, conn_mask)

    if estimator.is_success():
        # (mx, my), angle, conn_img, wire_img = estimator.result
        (mx, my), angle, conn_img = estimator.result ## 只分析 connector 的 pose，不檢查是否有線
        print(f"角度為 {angle:.2f}°，中點為 ({mx}, {my})")
        world_pos = image_to_world((mx, my), H_homo)
        print(f"世界座標為 {world_pos[0]}, {world_pos[1]}")
        cv2.imshow("Connector Result", conn_img)
        cv2.imwrite("./dataset/experiment/predict/7_result.png", conn_img)
        # cv2.imshow("Wire Result", wire_img)
    else:
        print("[!] 分析失敗，請檢查輸入圖像或格式")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
