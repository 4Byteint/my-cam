from picamera2 import Picamera2, Preview
import cv2
import os

# 初始化相機
picam2 = Picamera2()

camera_config = picam2.create_still_configuration(main={"size":(640,480)})  # 使用預覽模式
picam2.configure(camera_config)
# 關閉自動對焦(Af)，設置為手動模式
picam2.set_controls({"AfMode": 0, "LensPosition": 1.0})  # 固定焦距到 1m/10 = 10cm
picam2.set_controls({"AwbEnable": False, "ColourGains": (1.5, 0.7)})  # 1.7/0.7關掉白平衡，調整 Gain 值
picam2.set_controls({"ExposureValue": -0.5})  # +1 EV 提高亮度
picam2.start()

class ContactArea():
    def __init__(
        self, base=None, real_time=True,*args, **kwargs
    ):
        self.base = base
        self.real_time = real_time

    def __call__(self, target, base=None):
        base = self.base if base is None else base
        if base is None:
            raise AssertionError("A base sample must be specified for Pose.")
        base = self.crop_image(base)
        target = self.crop_image(target)
        target_copy = target.copy()
        diff = self._preprocess(base, target)
        output, center, angle = self._get_ellipse_contours(diff, target_copy)
        return output, center, angle

    def crop_image(self, image):
        x1, y1 = 210, 0  # 左上角
        x2, y2 = 462, 335  # 右下角
        return image[y1:y2, x1:x2]
    
    def _preprocess(self, base, target):
        diff = cv2.absdiff(base, target)
        sample_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blurred_diff = cv2.GaussianBlur(sample_gray, (7, 7), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 16))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 16))
        dilated = cv2.dilate(blurred_diff, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel2, iterations=1)
        ret,binary_im = cv2.threshold(eroded,25, 255,cv2.THRESH_BINARY)
        return binary_im
    
    def _get_ellipse_contours(self, binary_im, target_copy):
        contours,hierarchy = cv2.findContours(binary_im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key = cv2.contourArea)
            if len(max_contour) >= 50:  # fitEllipse requires at least 5 points
                ellipse = cv2.fitEllipse(max_contour)
                output = cv2.ellipse(target_copy, ellipse, (0, 255, 0), 2)
                
                # 打印橢圓的中心點和旋轉角度
                center, axes, angle = ellipse
                print(f"Ellipse Center: {center}")
                print(f"Ellipse Angle: {angle}")
                return target_copy, center, angle
        return target_copy, None, None

# ==============================================================
# main
def showRealtimeImage():
    base_path = './trans-processing/inference/2-RG/2-RG/img0-base.png'
    base_image = cv2.imread(base_path)
    if base_image is None:
        raise AssertionError("A base sample must be specified for Pose.")

    while True:
        # 獲取相機影像數據
        frame = picam2.capture_array()
        # 修正色彩空間（RGB -> BGR）
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame,0)
        contact_area=ContactArea(base=base_image,real_time=True)
        output, center, angle = contact_area(target=frame) # __call__ method
        cv2.imshow("frame", output)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
#####################################################
if __name__ == "__main__":
    showRealtimeImage()
    picam2.stop()
