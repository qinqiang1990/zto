import cv2 as cv
import numpy as np
import os
import random

# path = "1.jpg"
path = "../roi.jpg"


def detection(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # ret, dst = cv.threshold(gray, 200, 255, cv.THRESH_OTSU)
    ret, dst = cv.threshold(gray, 188, 255, cv.THRESH_BINARY_INV)
    return dst


image = cv.imread(path)

img = cv.pyrMeanShiftFiltering(src=image, sp=50, sr=50)

# dst = detection(img)

h, w = img.shape[:2]
mask = np.zeros([h + 2, w + 2], np.uint8)

for i in range(h):
    for j in range(w):
        # 非0处即为1，表示已经经过填充，不再处理
        if mask[i + 1, j + 1] == 0:
            cv.floodFill(img, mask, (j, i),
                         newVal=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                         loDiff=(20, 20, 20), upDiff=(20, 20, 20),
                         flags=8 | 1 << 8 | cv.FLOODFILL_FIXED_RANGE)

# src, contours, hierarchy = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
# cv.drawContours(image, contours, -1, (0, 0, 255), 2)
cv.namedWindow('img', cv.WINDOW_NORMAL)
cv.imshow('img', img)

cv.waitKey(0)
