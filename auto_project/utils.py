#def find_mask_centorid():
	#if mask=='wire':
		#print('wire coordinate: ')
		#if mask == 'connector':
			#print('conncetor centorid: ')

#def draw():
import cv2

def draw_fps(frame, fps):
	cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	return frame

def apply_perspective_transform(image, H):
    """
    對影像應用透視變換，將偵測到的四邊形轉換為標準矩形
    :param image: 原始影像
    :param sorted_points: 四個角點座標 (左上、右上、右下、左下)
    :return: 透視變換後的影像
    """
    # 應用透視變換
    warped_image = cv2.warpPerspective(image, H, config.PERSPECTIVE_SIZE)
    return warped_image
