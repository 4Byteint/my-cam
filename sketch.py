import cv2
import numpy as np

# 讀取影像
base_path = './trans-processing/none.png'
sample_path = './trans-processing/wire.png' 
base_image = cv2.imread(base_path)
sample_image = cv2.imread(sample_path)

##############################################################################################################
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
        contours = self._contours(diff, sample_image)
        # 如果無法計算出接觸面積，則拋出異常
        if self._compute_contact_area(contours, self.contour_threshold) == None and self.real_time==False:
            raise Exception("No contact area detected.")
        if self._compute_contact_area(contours, self.contour_threshold) == None and self.real_time==True:
            return None
        else:
            (
                poly,
                major_axis,
                major_axis_end,
                minor_axis,
                minor_axis_end,
                center,
                theta,
            ) = self._compute_contact_area(contours, self.contour_threshold)
        if self.draw_poly:
            self._draw_major_minor(
                target, poly, major_axis, major_axis_end, minor_axis, minor_axis_end
            )
        return center, theta, (major_axis, major_axis_end), (minor_axis, minor_axis_end)
        
    # 儲存影像
    def img_save(self, img, filename):
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
    def _contours(self, target, sample_image):
        # print(f"影像最小值: {target.min()}, 影像最大值: {target.max()}")
        mask = ((np.abs(target) > 0.025) * 255).astype(np.uint8) # 抓出變化的部分
        kernel = np.ones((16, 16), np.uint8)
        mask = cv2.erode(mask, kernel) # 侵蝕
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # 找輪廓
        # save contours
        draw_image = sample_image.copy()
        contours_img = cv2.drawContours(draw_image, contours, -1, (0, 0, 255), 1)  
        cv2.imwrite("contours_res.png", contours_img)
        return contours
    
    def _draw_major_minor(
        self,
        target,
        poly,
        
        major_axis,
        major_axis_end,
        minor_axis,
        minor_axis_end,
        lineThickness=2,
    ):
        # major_axis(red), minor_axis(green)
        cv2.polylines(target, [poly], True, (255, 255, 255), lineThickness)
        cv2.line(
            target,
            (int(major_axis_end[0]), int(major_axis_end[1])), 
            (int(major_axis[0]), int(major_axis[1])), 
            (0, 0, 255),
            lineThickness,
        )
        cv2.line(
            target,
            (int(minor_axis_end[0]), int(minor_axis_end[1])),
            (int(minor_axis[0]), int(minor_axis[1])),
            (0, 255, 0),
            lineThickness,
        )

    def _compute_contact_area(self, contours, contour_threshold):
        for contour in contours:
            if len(contour) > contour_threshold: # len(contour) 代表有多少個點組成的輪廓
                ellipse = cv2.fitEllipse(contour) # return: center, (major, minor)直徑, angle
                poly = cv2.ellipse2Poly(
                    (int(ellipse[0][0]), int(ellipse[0][1])),
                    (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), # axis length
                    int(ellipse[2]),
                    0,
                    360,
                    5, # 每5度取一次點
                )
                center = np.array([ellipse[0][0], ellipse[0][1]])
                a, b = (ellipse[1][0] / 2), (ellipse[1][1] / 2) # 長短軸
                theta = (ellipse[2] / 180.0) * np.pi # 角度轉弧度
                major_axis = np.array(
                    [center[0] - b * np.sin(theta), center[1] + b * np.cos(theta)]
                )
                minor_axis = np.array(
                    [center[0] + a * np.cos(theta), center[1] + a * np.sin(theta)]
                )
                major_axis_end = 2 * center - major_axis
                minor_axis_end = 2 * center - minor_axis
                return poly, major_axis, major_axis_end, minor_axis, minor_axis_end, center, theta

    
contact_area = contactArea(base=base_image)
center, theta, major, minor = contact_area(sample_image)
print("Major Axis: {0}, minor axis: {1}".format(*major, *minor))
print("center: {0}, angle(rad.): {1}".format(center, theta)) 
cv2.imwrite("contact_res.png", sample_image)