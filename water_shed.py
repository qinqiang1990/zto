import numpy as np
import cv2
import FFT
import telephone.common as common

img = cv2.imread('roi.jpg')
img = common.equalizeHist_(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -10)

area = common.Removing_small_connected_domain(thresh, 100)

cv2.imwrite("roi_.jpg", area)
area = common.dilate_(area, ksize=(5, 5))

h_left, h_right = common.border(area, axis=1)
w_left, w_right = common.border(area, axis=0)

cv2.imwrite("roi_1.jpg", gray[h_left:h_right, w_left:w_right])
