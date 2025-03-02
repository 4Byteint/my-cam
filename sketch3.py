import cv2
import numpy as np
 
def illum(image_path):
    img = cv2.imread(image_path)
    # img = img[532:768, 0:512]
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.threshold(img_bw, 90, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("img_thresh", img_thresh)
    cv2.waitKey(0)
    cnts = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    mask = np.zeros(img.shape, dtype=np.uint8)
    # img[thresh == 255] = 150
    for cnt in cnts:
        if len(cnt) > 0:
            x, y, w, h = cv2.boundingRect(cnt)
            mask[y:y+h, x:x+w] = 255
    cv2.imshow("mask", mask)
    cv2.imshow("mask", mask)
    result = cv2.illuminationChange(img, mask, alpha=1, beta=2)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    return result


image_path = "./trans-processing/trans-light-none.png"
illum(image_path)