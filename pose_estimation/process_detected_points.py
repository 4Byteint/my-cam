import os
import cv2
import numpy as np
from test_for_points import detect_circles
from coor.test_world_coor import backproject_to_camera

def load_calibration_data():
    """
    載入校準數據
    """
    here = os.path.dirname(__file__)               # …/camera/pose_estimation
    root = os.path.abspath(os.path.join(here, '..'))  # …/camera

    # 載入必要的矩陣
    H = np.load(os.path.join(root, "calibration", "perspective_matrix_128x160.npy")).astype(np.float32)
    mtx = np.load(os.path.join(root, "calibration", "camera_matrix.npy"))
    dist = np.load(os.path.join(root, "calibration", "dist_coeff.npy"))
    rvec = np.load(os.path.join(root, "calibration", "rvecs.npy"))
    tvec = np.load(os.path.join(root, "calibration", "tvecs.npy"))
    
    return H, mtx, dist, rvec, tvec

def process_detected_points(circles_info, mtx, H, roi_offset):
    """
    處理檢測到的點，計算其世界座標
    
    Args:
        circles_info: 檢測到的圓形信息列表，每個元素為 (x, y, r)
        mtx: 相機內參矩陣
        H: 透視變換矩陣
        roi_offset: ROI偏移量 (x, y)
    
    Returns:
        list: 每個點的處理結果，包含原始像素座標、射線方向和實際點座標
    """
    results = []
    for x, y, r in circles_info:
        # 轉換為整數座標
        u_pix, v_pix = int(x), int(y)
        
        # 計算世界座標
        ray_cam, real_point = backproject_to_camera(
            u_pix=u_pix,
            v_pix=v_pix,
            K=mtx,
            H=H,
            roi_offset=roi_offset
        )
        
        results.append({
            'pixel_coords': (x, y),
            'ray_direction': ray_cam,
            'world_coords': real_point,
            'radius': r
        })
    
    return results

def main():
    # 載入校準數據
    H, mtx, dist, rvec, tvec = load_calibration_data()
    
    # 設置圖片路徑
    here = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(here, '..'))
    background_path = os.path.join(root, "calibration", "demo", "transform", "img1_points.png")
    foreground_path = os.path.join(root, "calibration", "demo", "transform", "img2_points.png")
    
    # 檢測圓形
    output_image, circles_info = detect_circles(background_path, foreground_path)
    
    if output_image is not None and circles_info:
        # 顯示檢測結果
        cv2.imshow("Detected Circles", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 處理檢測到的點
        roi_offset = (156, 29)  # ROI後左上角的座標
        results = process_detected_points(circles_info, mtx, H, roi_offset)
        
        # 打印結果
        print("\n檢測到的點及其相機座標：")
        print("-" * 50)
        for i, result in enumerate(results, 1):
            print(f"\n點 {i}:")
            print(f"像素座標: ({result['pixel_coords'][0]:.3f}, {result['pixel_coords'][1]:.3f})")
            print(f"半徑: {result['radius']:.3f}")
            print(f"射線方向: {result['ray_direction']}")
            print(f"世界座標: {result['world_coords']}")
            print("-" * 50)
        
        return results
    else:
        print("未檢測到任何圓形或圖片讀取失敗")
        return None

if __name__ == "__main__":
    main() 