import os
import numpy as np
import cv2

def print_camera_matrix():
    """
    讀取並打印相機內參矩陣和外部參數的詳細資訊
    """
    # 獲取檔案路徑
    here = os.path.dirname(__file__)               # …/camera/pose_estimation/coor
    root = os.path.abspath(os.path.join(here, '../../'))  # …/camera
    mtx_path = os.path.join(root, "calibration", "camera_matrix.npy")
    rvecs_path = os.path.join(root, "calibration", "rvecs.npy")
    tvecs_path = os.path.join(root, "calibration", "tvecs.npy")
    
    # 檢查檔案是否存在
    if not all(os.path.exists(p) for p in [mtx_path, rvecs_path, tvecs_path]):
        print("錯誤：找不到必要的校準檔案")
        return
    
    try:
        # 載入相機矩陣
        mtx = np.load(mtx_path)
        rvecs = np.load(rvecs_path)
        tvecs = np.load(tvecs_path)
        
        print("\n=== 相機內參矩陣資訊 ===")
        print(f"檔案路徑：{mtx_path}")
        print(f"矩陣形狀：{mtx.shape}")
        print("\n矩陣內容：")
        print(f"fx = {mtx[0,0]:.2f} (x方向焦距)")
        print(f"fy = {mtx[1,1]:.2f} (y方向焦距)")
        print(f"cx = {mtx[0,2]:.2f} (x方向主點)")
        print(f"cy = {mtx[1,2]:.2f} (y方向主點)")
        print("\n完整矩陣：")
        print(mtx)
        
        # 計算其他相關資訊
        print("\n其他資訊：")
        print(f"x方向焦距（像素）：{mtx[0,0]:.2f}")
        print(f"y方向焦距（像素）：{mtx[1,1]:.2f}")
        print(f"主點位置：(cx, cy) = ({mtx[0,2]:.2f}, {mtx[1,2]:.2f})")
        
        # 檢查焦距是否相等
        if np.isclose(mtx[0,0], mtx[1,1], rtol=1e-3):
            print("\n注意：x和y方向的焦距非常接近，可以視為相等")
        else:
            print("\n警告：x和y方向的焦距有明顯差異")
            print(f"差異：{abs(mtx[0,0] - mtx[1,1]):.2f} 像素")
        
        print("\n=== 相機外部參數資訊 ===")
        print(f"旋轉向量檔案：{rvecs_path}")
        print(f"平移向量檔案：{tvecs_path}")
        
        # 顯示外部參數
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            print(f"\n第 {i+1} 組外部參數：")
            print(f"旋轉向量 (rvec)：{rvec.flatten()}")
            print(f"平移向量 (tvec)：{tvec.flatten()}")
            
            # 將旋轉向量轉換為旋轉矩陣
            R, _ = cv2.Rodrigues(rvec)
            print("\n旋轉矩陣：")
            print(R)
            
            # 計算歐拉角（以度為單位）
            euler_angles = np.degrees(cv2.RQDecomp3x3(R)[0])
            print("\n歐拉角（度）：")
            print(f"繞 X 軸旋轉：{euler_angles[0]:.2f}°")
            print(f"繞 Y 軸旋轉：{euler_angles[1]:.2f}°")
            print(f"繞 Z 軸旋轉：{euler_angles[2]:.2f}°")
            
    except Exception as e:
        print(f"讀取檔案時發生錯誤：{str(e)}")

if __name__ == "__main__":
    print_camera_matrix() 