import cv2
import numpy as np

# 讀取影像
base_path = './trans-processing/none.png'
sample_path = './trans-processing/wire.png' 
base_image = cv2.imread(base_path)
sample_image = cv2.imread(sample_path)

##############################################################################################################
# resized = cv2.resize(diff, (640, 480))
# denoised = cv2.medianBlur(resized, 3)  # 核大小可調，如 3、5、7
# cv2.imwrite("denoised.png", denoised)
# # 二值化再取輪廓
# _, binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
# cv2.imwrite("binary.png", binary)
# contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# original_image = sample_image.copy()
# output = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
# cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

class contactArea():
    # 初始化 & 設定呼叫函數
    def __init__(
        self, base=None, draw_poly=True, contour_threshold=100,real_time=False,*args, **kwargs
    ):
        self.base = base
        self.draw_poly = draw_poly
        self.contour_threshold = contour_threshold
        self.real_time = real_time

    def __call__(self, target, base=None):
        base = self.base if base is None else base
        if base is None:
            raise AssertionError("A base sample must be specified for Pose.")
        diff = self._diff(target, base)
        self.img_save(diff, "diff")
        diff = self._smooth(diff)
        self.img_save(diff, "smooth")
        contours = self._contours(diff)
##############################################################################################################
    # 儲存影像
    def img_save(self, img, filename):
        print(img.dtype)
        img = (img * 255).astype(np.uint8)
        cv2.imwrite(f"{filename}_res.png", img)
    # 差分
    def _diff(self, target, base):
        # 把正規化到[-0.5,1.5]範圍轉換到中心為0.5，並且避免負數，如果target==base,diff=0.5
        diff = (target * 1.0 - base) / 255.0 + 0.5
        # 對diff非線性壓縮 target<0.5,把target比base暗的區域壓縮，更靠近0.5，減少陰影部份的變化
        diff[diff < 0.5] = (diff[diff < 0.5] - 0.5) * 0.7 + 0.5 # *0.7縮小 +0.5 恢復範圍
        # 計算diff相對0.5的偏移量取絕對值 axis=-1最後一個維度計算
        diff_abs = np.mean(np.abs(diff - 0.5), axis=-1)
        # target和base的變異程度 # 如果是浮點數，需轉成 0-255 範圍
        return diff_abs
    # 平滑化
    def _smooth(self, target):
        kernel = np.ones((16, 16), np.float32) 
        # 確保 kernel 總和=1 不會改變影像的平均亮度
        kernel /= kernel.sum()
        diff_blur = cv2.filter2D(target, -1, kernel) # 平滑化
        return diff_blur
    # 提取輪廓
    def _contours(self, target):
        print(f"target min: {target.min()}, target max: {target.max()}")
        mask = ((np.abs(target) > 0.025) * 255).astype(np.uint8) # 抓出變化的部分
        kernel = np.ones((16, 16), np.uint8)
        mask = cv2.erode(mask, kernel) # 侵蝕
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # 找輪廓
        contours_img = cv2.drawContours(mask, contours, -1, (255, 255, 255), 1)  # 白色輪廓，線寬為1
        cv2.imwrite("contours_res.png", contours_img)
        # print(f"mask min: {mask.min()}, mask max: {mask.max()}")
        return contours
    
contact_area = contactArea(base=base_image)
contact_area(sample_image)
