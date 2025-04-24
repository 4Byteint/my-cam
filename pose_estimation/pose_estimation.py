import math
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def find_edge_points(img_path):
    img = cv2.imread(img_path)
    # 轉為灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 取得圖像寬高
    height, width = gray.shape
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 取最大輪廓（通常是白色區塊）
    cnt = max(contours, key=cv2.contourArea)
    # 多邊形擬合，取得邊的頂點
    epsilon = 0.1 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    # 閾值（誤差）
    x_thresh = 10
    y_thresh = 10
    y_target = height
    # # 逐邊畫出邊線（擬合線段）
    # for i in range(len(approx)):
    #     pt1 = tuple(approx[i][0])
    #     pt2 = tuple(approx[(i+1) % len(approx)][0])  # 閉合輪廓
    #      # 過濾邊：排除 x ≈ 0 的邊（左右端點都接近 x=0）
    #     if abs(pt1[0]) <= x_thresh and abs(pt2[0]) <= x_thresh:
    #         continue
    #     # 排除：右邊框（x ≈ image width）
    #     if abs(pt1[0] - width) <= x_thresh and abs(pt2[0] - width) <= x_thresh:
    #         continue
    #     # 過濾邊：排除 y ≈ 160 的邊（上下端點都接近 y=160）
    #     if abs(pt1[1] - y_target) <= y_thresh and abs(pt2[1] - y_target) <= y_thresh:
    #         continue
    #     # 畫出這一條邊
    #     cv2.line(img, pt1, pt2, (0, 255, 0), 2)  # 用藍色畫出邊
    #     edges.append((pt1, pt2))

    # Step 1：過濾角點（剔除邊界角點）
    filtered_pts = []
    for pt in approx:
        x, y = pt[0]
        if abs(x - 0) <= x_thresh:
            continue
        if abs(x - width) <= x_thresh:
            continue
        if abs(y - y_target) <= y_thresh:
            continue
        filtered_pts.append((x, y))

    # Step 2：畫線、畫中點、畫垂線
    L = 40  # 垂線長度
    if len(filtered_pts) == 2:
        pt1 = filtered_pts[0]
        pt2 = filtered_pts[1]  # 若不想閉合，可改為 i+1 檢查邊界

        # 畫邊（綠色）
        cv2.line(img, pt1, pt2, (0, 255, 0), 2)

        # 計算中心點
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2
        center = (mid_x, mid_y)
        cv2.circle(img, center, 4, (0, 255, 255), -1)  # 中點黃色

        # 邊向量與長度
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = math.hypot(dx, dy)
    

        # 垂直向量（單位向量）
        perp_dx = -dy / length
        perp_dy = dx / length

        # 垂線兩端點
        pA = (int(mid_x + perp_dx * L), int(mid_y + perp_dy * L))
        pB = (int(mid_x - perp_dx * L), int(mid_y - perp_dy * L))
        cv2.line(img, pA, pB, (0, 0, 255), 2)  # 紅色垂直線

        # 計算垂線與 x 軸夾角
        angle_rad = math.atan2(perp_dy, perp_dx)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0]), int(pt2[1]))
        # 印出資訊
        print(f"  pt1 = {pt1}, pt2 = {pt2}")
        print(f"  中點 = {center}")
        print(f"  邊長 = {length:.2f}")
        print(f"  垂線與 x 軸夾角 = {angle_deg:.2f}°\n")
    # 顯示結果
    cv2.imshow("All edges", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# def find_edge_points_from_video(video_path):
# def find_pose_form_line():
    
########################################################
def main():
    # 載入新的圖像
    image_path = "C:/Jill/Code/camera/model_train/predict_final/img50_predict_connector.png"
    find_edge_points(image_path)


if __name__ == "__main__":
    main()
