# coding:utf-8
import sys
import os
import cv2
from telephone import common

file_name = "./dataset/216323577051.jpg"

# 读取图片
origin = cv2.imread(file_name)

# 灰度化
gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)

# 二值化
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -10)

img = binary

# 中值滤波
img = cv2.medianBlur(img, 3)

# img = common.Removing_small_connected_domain(img, 30)

# 形态学处理
img = common.dilate_(img, ksize=(3, 3), iterations=5)
img = common.morph_(img, ksize=(50, 50))

# 轮廓提取
img, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (255, 0, 0), 3)

# 分割轮廓
cut_binary = common.minAreaRect_(binary, contours)

img=cut_binary
# img = common.hough_lines_(img, cut_binary)

print("image shape:", img.shape)
# 显示框大小
# window_size = img.shape
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.moveWindow("image", int(window_size[1] / 50), int(window_size[0] / 50))
# cv2.resizeWindow("image", int(window_size[1] / 5), int(window_size[0] / 5))
cv2.imshow("image", img)
cv2.waitKey()
cv2.destroyAllWindows()
