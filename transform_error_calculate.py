import numpy as np
import cv2
from typing import List, Tuple

import config

def calculate_homography(image_points: List[List[float]], world_points: List[List[float]]) -> np.ndarray:
    """
    計算單應性矩陣
    
    Args:
        image_points: 圖像座標點列表 [[x1,y1], [x2,y2], ...]
        world_points: 世界座標點列表 [[x1,y1], [x2,y2], ...]
    
    Returns:
        單應性矩陣
    """
    src_points = np.array(image_points, dtype=np.float32)
    dst_points = np.array(world_points, dtype=np.float32)
    H_homo, _ = cv2.findHomography(src_points, dst_points)
    return H_homo

def image_to_world(image_point: List[float], H_homo: np.ndarray) -> Tuple[float, float]:
    """
    將圖像座標轉換為世界座標
    
    Args:
        image_point: 圖像座標點 [x, y]
        H: 單應性矩陣
    
    Returns:
        世界座標點 (x, y)
    """
    point = np.array([[image_point]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_point = cv2.perspectiveTransform(point, H_homo)
    return transformed_point[0][0][0], transformed_point[0][0][1]

def calculate_error(predicted: Tuple[float, float], ground_truth: Tuple[float, float]) -> Tuple[float, float, float]:
    """
    計算預測點與實際點之間的誤差
    
    Args:
        predicted: 預測的世界座標點 (x, y)
        ground_truth: 實際的世界座標點 (x, y)
    
    Returns:
        (x方向誤差, y方向誤差, 總誤差)
    """
    x_error = abs(predicted[0] - ground_truth[0])
    y_error = abs(predicted[1] - ground_truth[1])
    total_error = np.sqrt(x_error**2 + y_error**2)
    return x_error, y_error, total_error

def calculate_center_point(points: List[List[float]]) -> List[float]:
    """
    計算四邊形的中心點
    
    Args:
        points: 四個點的列表 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    
    Returns:
        中心點座標 [x, y]
    """
    x_sum = sum(point[0] for point in points)
    y_sum = sum(point[1] for point in points)
    return [x_sum / 4, y_sum / 4]

def calculate_edge_center(points: List[List[float]], edge: str) -> List[float]:
    """
    計算四邊形邊的中心點
    
    Args:
        points: 四個點的列表 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        edge: 邊的位置 ('top', 'bottom', 'left', 'right')
    
    Returns:
        邊的中心點座標 [x, y]
    """
    if edge == 'top':
        return [(points[0][0] + points[1][0])/2, (points[0][1] + points[1][1])/2]
    elif edge == 'bottom':
        return [(points[2][0] + points[3][0])/2, (points[2][1] + points[3][1])/2]
    elif edge == 'left':
        return [(points[0][0] + points[3][0])/2, (points[0][1] + points[3][1])/2]
    elif edge == 'right':
        return [(points[1][0] + points[2][0])/2, (points[1][1] + points[2][1])/2]
    else:
        raise ValueError("edge must be one of: 'top', 'bottom', 'left', 'right'")


def generate_square(x, y, size=5.0):
    return [
        (x, y),
        (x + size, y),
        (x + size, y - size),
        (x, y - size),
    ]

# 示例使用
if __name__ == "__main__":
    world_ref_pts = [
        (-9, 45.5),
        (9, 45.5),
        (9, 23.5),
        (-9, 23.5)
    ]
    # 上下左右中心點
    world_center_pts = [
        (0.0, 45.5),
        (0.0, 23.5),
        (-9, 34.5),
        (9, 34.5),
        (0.0, 34.5)
    ]
    world_square_origins = [
        (-7.0, 41.5),
        (2, 41.5),
        (-7, 32.5),
        (2, 32.5)
    ]
    world_square_pts = [generate_square(x, y) for x, y in world_square_origins]
    
    # 圖像座標點
    image_pts = [
        (0, 0),
        (180, 0),
        (180, 220),
        (0, 220)
    ]
    image_square_pts = config.SQUARE_POINTS # from transformed image
    
    # 透射變換
    # H = np.load(config.PERSPECTIVE_MATRIX_PATH).astype(np.float32)
    # 計算單應性矩陣
    H_homo = calculate_homography(image_pts, world_ref_pts)
    
    # 計算各個位置的中心點
    center_point = calculate_center_point(image_pts)
    top_center = calculate_edge_center(image_pts, 'top')
    bottom_center = calculate_edge_center(image_pts, 'bottom')
    left_center = calculate_edge_center(image_pts, 'left')
    right_center = calculate_edge_center(image_pts, 'right')
    
    # 測試所有中心點
    test_points = {
        "上邊中心": (top_center, world_center_pts[0]),
        "下邊中心": (bottom_center, world_center_pts[1]),
        "左邊中心": (left_center, world_center_pts[2]),
        "右邊中心": (right_center, world_center_pts[3]),
        "中心點": (center_point, world_center_pts[4]),
        "正方形1": (image_square_pts[0], world_square_pts[0]),
        "正方形2": (image_square_pts[1], world_square_pts[1]),
        "正方形3": (image_square_pts[2], world_square_pts[2]),
        "正方形4": (image_square_pts[3], world_square_pts[3]),
    }
    
    # 對每個測試點進行轉換和誤差計算
    for point_name, (test_point, ground_truth) in test_points.items():
        if point_name.startswith("正方形"):
        # 跳過正方形群組，另行處理
            continue
        print(f"\n測試 {point_name}:")
        print(f"圖像座標: {test_point}")
        predicted_world_point = image_to_world(test_point, H_homo)
        print(f"預測的世界座標點: ({predicted_world_point[0]:.3f}, {predicted_world_point[1]:.3f})")
        print(f"實際世界座標點: ({ground_truth})")
        
        x_error, y_error, total_error = calculate_error(predicted_world_point, ground_truth)
        print(f"X方向誤差: {x_error:.3f} mm")
        print(f"Y方向誤差: {y_error:.3f} mm")
        print(f"總誤差: {total_error:.3f} mm")
        print("--------------------------------")
    # 群組正方形點測試
    for i in range(4):  # 正方形1~4
        image_pts = image_square_pts[i]      # 圖像座標點（4個）
        world_pts = world_square_pts[i]      # 對應的世界座標點（4個）
        
        print(f"\n測試 正方形{i+1}:")
        
        for j, (img_pt, world_pt) in enumerate(zip(image_pts, world_pts)):
            predicted_world_point = image_to_world(img_pt, H_homo)
            x_error, y_error, total_error = calculate_error(predicted_world_point, world_pt)
            
            print(f"  點 {j+1}:")
            print(f"    圖像座標: {img_pt}")
            print(f"    預測世界座標: ({predicted_world_point[0]:.3f}, {predicted_world_point[1]:.3f})")
            print(f"    實際世界座標: ({world_pt[0]:.3f}, {world_pt[1]:.3f})")
            print(f"    X誤差: {x_error:.3f} mm, Y誤差: {y_error:.3f} mm, 總誤差: {total_error:.3f} mm")
        print("--------------------------------")
