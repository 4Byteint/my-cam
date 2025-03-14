import cv2
import numpy as np

def crop_trapezoid_with_bounded_mask(image_path, points):
    """
    使用遮罩來保留梯形區域，其他部分變黑，且輸出影像大小為梯形的包圍矩形大小
    :param image_path: 影像檔案路徑
    :param points: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] 梯形四個點的座標
    :return: 只保留梯形區域的影像，大小與梯形包圍矩形一致
    """
    image = cv2.imread(image_path)
    # 轉換為 NumPy 陣列
    pts = np.array(points, dtype=np.int32)

    # 計算梯形的包圍矩形 (bounding box)
    x, y, w, h = cv2.boundingRect(pts)  # (x, y) 為左上角座標，(w, h) 為寬高

    # 創建與梯形包圍矩形大小相同的黑色遮罩
    mask = np.zeros((h, w), dtype=np.uint8)

    # 調整梯形座標，使其適應新的遮罩大小
    pts_adjusted = pts - [x, y]  # 讓 (x, y) 為 (0,0)

    # 在遮罩上填充白色，表示我們想要保留的區域
    cv2.fillPoly(mask, [pts_adjusted], 255)

    # 裁剪原始影像，使大小與 bounding box 相同
    cropped_image = image[y:y+h, x:x+w]

    # 只保留梯形區域，其他部分變黑
    masked_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)

    return masked_image


# 測試用：設定梯形四個點的座標 (左上、右上、右下、左下)
image_path = "./imprint/img0_base.png"  # 替換為你的影像檔案
trapezoid_points = [(136, 0), (515, 0), (458, 340), (200, 348)]  # 替換為你的梯形座標
cropped_image = crop_trapezoid_with_bounded_mask(image_path, trapezoid_points)

if cropped_image is not None:
    cv2.imshow("Masked Trapezoid", cropped_image)
    cv2.imwrite("./imprint/al/cropped/img0_base.png", cropped_image)
    print("shape_size: ", cropped_image.shape)
    cv2.waitKey(0)
    cv2.destroyAllWindows()