import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_mpp(image_path, chessboard_size, square_size, camera_matrix=None, dist_coeffs=None):
    """
    使用棋盤格標定計算 mpp
    :param image_path: 棋盤格影像
    :param chessboard_size: 棋盤格內部角點數量 (cols, rows)
    :param square_size: 每個方格的真實尺寸 (mm)
    :return: 計算出的 mpp (mm/pixel)
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 若提供了內部參數與畸變係數，先進行去畸變處理
    if camera_matrix is not None and dist_coeffs is not None:
        gray = cv2.undistort(gray, camera_matrix, dist_coeffs)

    # 偵測棋盤格角點 (初步偵測)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if not ret:
        print("未偵測到棋盤格，請確認影像")
        return None

    # 設定亞像素級細化參數
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)  # (類型, 最大迭代次數, 精度閾值)
    
    # 使用 cornerSubPix 進一步細化角點
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # 取出角點座標
    corners = corners.squeeze()  # (N, 1, 2) -> (N, 2)

    # 計算 X 方向的 mpp
    num_cols = chessboard_size[0] - 1  # 內部角點數量
    x_dists = [
        np.linalg.norm(corners[i] - corners[i + 1])
        for i in range(num_cols)
    ]
    avg_x_dist = np.mean(x_dists)  # 計算像素距離
    mpp_x = square_size / avg_x_dist  # mm/pixel

    # 計算 Y 方向的 mpp
    num_rows = chessboard_size[1] - 1
    y_dists = [
        np.linalg.norm(corners[i] - corners[i + chessboard_size[0]])
        for i in range(0, num_rows * chessboard_size[0], chessboard_size[0])
    ]
    avg_y_dist = np.mean(y_dists)
    mpp_y = square_size / avg_y_dist

    print(f"mpp (X方向): {mpp_x:.6f} mm/pixel")
    print(f"mpp (Y方向): {mpp_y:.6f} mm/pixel")
    return (mpp_x + mpp_y) / 2  # 取平均值作為 mpp

def detect_sphere_imprints(base_path, sample_path):
    """
    使用霍夫圓變換偵測球體壓痕
    :param image_path: 壓痕影像的路徑
    :param min_radius: 可調整的最小半徑（像素）
    :param max_radius: 可調整的最大半徑（像素）
    :return: 壓痕的圓心座標 (x, y) 和半徑 r
    """
    # 讀取影像並轉換為灰階
    base_img = cv2.imread(base_path)
    sample_img = cv2.imread(sample_path)
    diff = cv2.absdiff(base_img, sample_img)
    # cv2.imshow("diff", diff)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # 進行高斯模糊，減少雜訊影響
    # cv2.imshow("gray", gray)
    
    # 使用 CLAHE 增強局部對比
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray)
    # cv2.imshow("enhanced_img", enhanced_img)
    gray_blurred = cv2.GaussianBlur(gray, (5,5), 2)
    # cv2.imshow("gray_blurred", gray_blurred)
    # cv2.waitKey(1000)  # 等待1秒
    # cv2.destroyAllWindows()
    # 使用霍夫圓變換來偵測圓形壓痕
    # miniDist: 
    # param1: canny thershold
    # param2: hough 累積的 thershold 越小檢測到的圓越多
    circles = cv2.HoughCircles(enhanced_img, 
                               cv2.HOUGH_GRADIENT, 
                               dp=1.2, 
                               minDist=100,
                               param1=60, 
                               param2=30, 
                               minRadius=15, 
                               maxRadius=32)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        best_circle = max(circles[0, :], key=lambda c: c[2])  # 取半徑最大的圓
        for circle in circles[0, :]:
            x, y, r = circle
            # 在原圖上畫出所有圓
            cv2.circle(sample_img, (x, y), r, (0, 255, 0), 2) # 綠色圓圈
            cv2.circle(sample_img, (x, y), 2, (0, 0, 255), 3)  
        # 在原圖上畫出最大的圓
        x, y, r = best_circle
        cv2.circle(sample_img, (x, y), r, (255, 0, 0), 2) # 藍色圓圈
        cv2.circle(sample_img, (x, y), 2, (0, 0, 255), 3)  
        print(f"偵測到圓形壓痕: 圓心 ({x}, {y})，半徑 {r} 像素")
        # cv2.imshow("Detected Circles", sample_img)
        
        # 保存帶有圓形標記的圖片
        output_dir = os.path.dirname(sample_path)
        output_filename = os.path.basename(sample_path)
        output_path = os.path.join(output_dir, "circles", output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, sample_img)
        print(f"已保存帶有圓形標記的圖片：{output_path}")
        
        # cv2.waitKey(1000)  # 等待1秒
        # cv2.destroyAllWindows()
        return x, y, r
    else:
        print("未偵測到圓形壓痕，請確認影像品質")
        return None, None, None

def compute_surface_gradients(cx, cy, r, img_shape):
    """
    計算半圓球區域的表面梯度 (Gx, Gy)，忽略背景
    :param cx, cy: 圓心座標
    :param r: 球體半徑（像素）
    :param img_shape: 影像尺寸 (H, W)
    :return: 梯度場 (Gx, Gy) 和球體遮罩 mask
    """
    H, W = img_shape[:2]
    X, Y = np.meshgrid(np.arange(W) , np.arange(H) )
  
    # 轉換座標，使 Xc, Yc 相對於圓心
    Xc = X - cx
    Yc = Y - cy
    # 建立遮罩，只取半球內的像素
    mask = (Xc**2 + Yc**2 < r**2)
    # 計算球體 Z 值，確保 Z > 0（半圓球區域）
    Z = np.sqrt(np.maximum(r**2 - Xc**2 - Yc**2, 0) + 1e-12)  # 避免 sqrt 負數
   
    # 計算表面梯度，只對半圓球區域有效
    Gx = np.where(mask, Xc / Z, 0)
    Gy = np.where(mask, Yc / Z, 0)
    # 正規化梯度
    magnitude = np.sqrt(Gx**2 + Gy**2)
    Gx_norm = Gx / (magnitude + 1e-6)  # 避免除以零
    Gy_norm = Gy / (magnitude + 1e-6)
    
    # 打印梯度場的範圍
    print("Z min:", np.min(Z))
    print("Z max:", np.max(Z))
    print("X 範圍:", np.min(X), np.max(X))
    print("Y 範圍:", np.min(Y), np.max(Y))
    print("正規化後的 Gx 範圍:", np.min(Gx_norm), np.max(Gx_norm))
    print("正規化後的 Gy 範圍:", np.min(Gy_norm), np.max(Gy_norm))
    return Gx_norm, Gy_norm, mask
    


def lookup_table_dict(sample_path, Gx, Gy, image_id):
    import cv2
    import numpy as np

    img = cv2.imread(sample_path)
    H, W = img.shape[:2]

    # 擷取 RGB 通道
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # 建立座標網格
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    data = {
        'image_id': image_id,
        'x': X.flatten(),
        'y': Y.flatten(),
        'R': R.flatten(),
        'G': G.flatten(),
        'B': B.flatten(),
        'Gx': Gx.flatten(),
        'Gy': Gy.flatten()
    }
    return pd.DataFrame(data)

def visualize_gradient_heatmap(Gx, Gy):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax[0].imshow(Gx, cmap='coolwarm')
    ax[0].set_title("Gx Heatmap")
    plt.colorbar(im1, ax=ax[0])
    im2 = ax[1].imshow(Gy, cmap='coolwarm')
    ax[1].set_title("Gy Heatmap")
    plt.colorbar(im2, ax=ax[1])
    plt.show(block=False)  # 设置为非阻塞模式
    plt.pause(2)  # 显示2秒
    plt.close()

def query_lut(lut, x_query, y_query):
    result = lut[(lut['x'] == x_query) & (lut['y'] == y_query)]
    if result.size > 0:
        print(f"查詢座標 ({x_query}, {y_query})")
        print(f"R: {result['R'][0]}")
        print(f"G: {result['G'][0]}")
        print(f"B: {result['B'][0]}")
        print(f"Gx: {result['Gx'][0]:.10f}")
        print(f"Gy: {result['Gy'][0]:.10f}")
    else:
        print("未找到對應座標")
        
def save_lut_csv(lut, filename="lookup_table.csv"):
    df = pd.DataFrame(lut)
    df.to_csv(filename, index=False)
    print(f"LUT 已儲存為 {filename}")
def visualize_normals_on_image(Gx, Gy, original_img, step=2):
    """
    在原始影像上可視化正規化的梯度向量
    :param Gx: X方向梯度
    :param Gy: Y方向梯度
    :param original_img: 原始影像
    :param step: 箭頭間隔（像素）
    """
    # 創建箭頭圖
    H, W = original_img.shape[:2]
    X, Y = np.meshgrid(np.arange(0, W, step), np.arange(0, H, step))
    
    # 繪製箭頭
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.quiver(X, Y, Gx[::step, ::step], -Gy[::step, ::step],  # quiver使用的是數學座標系，影像坐標系往下+y
              color='red', scale=20, width=0.002)
    plt.title('Normalized Gradient Vectors')
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    
    
    
def create_rgb2gradient_dataset(input_dir, output_dir):
    """
    处理文件夹中的所有图片（排除包含 'base' 的图片），生成 RGB 到梯度的数据集
    :param input_dir: 输入文件夹路径
    :param output_dir: 输出文件夹路径（用于存储检测到的圆形）
    :return: 包含所有图片数据的 DataFrame
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储所有图片的数据
    all_data = []
    
    # 获取 base 图片
    base_file = None
    for f in os.listdir(input_dir):
        if f.lower().endswith('.png') and 'base' in f.lower():
            base_file = f
            break
    
    if base_file is None:
        print("未找到 base 图片（文件名需包含 'base'）")
        return
    
    base_path = os.path.join(input_dir, base_file)
    base_img = cv2.imread(base_path)
    
    if base_img is None:
        print(f"无法读取 base 图：{base_path}")
        return
    
    # 处理所有 sample 图片
    for sample_file in os.listdir(input_dir):
        if not sample_file.lower().endswith('.png'):
            continue
        if 'base' in sample_file.lower():
            continue
            
        # 从文件名中提取数字作为 image_id
        try:
            image_id = int(''.join(filter(str.isdigit, sample_file)))
        except:
            print(f"无法从文件名 {sample_file} 提取数字作为 image_id")
            continue
            
        sample_path = os.path.join(input_dir, sample_file)
        sample_img = cv2.imread(sample_path)
        if sample_img is None:
            print(f"无法读取：{sample_file}")
            continue
            
        print(f"处理图片：{sample_file}")
        
        # 检测球体压痕
        cx, cy, r = detect_sphere_imprints(base_path, sample_path)
        if cx is None or cy is None or r is None:
            print(f"无法检测到圆形压痕：{sample_file}")
            continue
            
        # 计算梯度
        Gx, Gy, mask = compute_surface_gradients(cx, cy, r, sample_img.shape)
        #visualize_normals_on_image(Gx, Gy, sample_img, step=2)
            
        # 创建当前图片的 lookup table
        lut = lookup_table_dict(sample_path, Gx, Gy, image_id)
        
        # 将数据添加到总数据中
        all_data.append(lut)
        
    
    #合并所有数据
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        # 保存为 CSV 文件
        output_csv = os.path.join(output_dir, "lookup_table.csv")
        final_df.to_csv(output_csv, index=False)
        print(f"数据集已保存为：{output_csv}")
        return final_df
    else:
        print("没有成功处理任何图片")
        return None



# 使用示例
input_folder = './imprint/al_RGB_calib/transform/'
output_folder = './imprint/al_RGB_calib/transform/circles/'
dataset = create_rgb2gradient_dataset(input_folder, output_folder)