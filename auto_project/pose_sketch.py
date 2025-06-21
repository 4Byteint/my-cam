import cv2
import numpy as np

def sketch_line(conn_mask, conn_bgr, topk=3):
    edges = cv2.Canny(conn_mask, 50, 150)
  
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30, # 至少多少算一條線
        minLineLength=10, # 至少多少像素算一條線
        maxLineGap=30 # 最大間隔多少像素算一條線
    )
    
    
    area = np.count_nonzero(conn_mask == 255)
    print(f"wire area: {area}")
    if area < 1000:
        print(f"[!] wire 白色區域面積小於 1000 像素，不顯示圖像")
        return False
    rows, cols = np.where(conn_mask == 255)   # 取得所有 white pixel 的 (row, col) 座標
    r_centroid = rows.mean()
    c_centroid = cols.mean()
    center = (int(c_centroid), int(r_centroid))
    cv2.circle(conn_bgr, center, 5, (0, 0, 255), -1)
        

    # 計算每條線段中點距離中心
    dist_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            midpoint = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            dist = np.linalg.norm(midpoint - center)
            dist_lines.append((dist, (x1, y1, x2, y2)))
    else: 
        print("[!] 找不到線")
    
    
      # 根據距離排序，取前 k 大
    dist_lines.sort(key=lambda x: x[0], reverse=True)
    top_lines = [item[1] for item in dist_lines[:topk]]

    # 畫出最外層線段
    for i, (x1, y1, x2, y2) in enumerate(top_lines):
        cv2.line(conn_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        

    return conn_bgr



def main():
    # 載入新的圖像（這裡是二值圖，應該為0/255）
    image_path = "./dataset/experiment/predict/img_color_connector.png"
    conn_mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)    

    if conn_mask is None:
        print(f"[!] 無法讀取影像：{image_path}")
        return
    
    # 轉成彩圖以利畫紅線
    conn_bgr = cv2.cvtColor(conn_mask, cv2.COLOR_GRAY2BGR)

    # 畫線
    conn_bgr = sketch_line(conn_mask, conn_bgr, topk=4)

    # 顯示結果
    cv2.imshow("Connector Result", conn_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
