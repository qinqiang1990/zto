# coding:utf-8

import cv2
import numpy as np

shrink = 0.3

img = cv2.imread("C:/Users/qinq/Pictures/zto/7.jpg")
print("img size:", img.shape)

height, width = img.shape[:2]
size = (int(width * shrink), int(height * shrink))

img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
print("img size:", img.shape)

x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
print(sobel.shape)
cv2.imshow("sobel", sobel[:,:,0])

gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
laplacian = cv2.convertScaleAbs(gray_lap)
cv2.imshow('laplacian', laplacian)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
canny = cv2.Canny(img, 50, 150, apertureSize=3)
print(canny.shape)
cv2.imshow('canny', canny)


lines = cv2.HoughLines(canny, 1, np.pi / 180, 118)  # 这里对最后一个参数使用了经验型的值
hough = img.copy()

for line in lines[0]:
    rho = line[0]  # 第一个元素是距离rho
    theta = line[1]  # 第二个元素是角度theta
    if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
        # 该直线与第一行的交点
        pt1 = (int(rho / np.cos(theta)), 0)
        # 该直线与最后一行的焦点
        pt2 = (int((rho - hough.shape[0] * np.sin(theta)) / np.cos(theta)), hough.shape[0])
        # 绘制一条白线
        cv2.line(hough, pt1, pt2, (255))
    else:  # 水平直线
        # 该直线与第一列的交点
        pt1 = (0, int(rho / np.sin(theta)))
        # 该直线与最后一列的交点
        pt2 = (hough.shape[1], int((rho - hough.shape[1] * np.cos(theta)) / np.sin(theta)))
        # 绘制一条直线
        cv2.line(hough, pt1, pt2, (255), 1)
cv2.imshow('Hough', hough)

lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 80, minLineLength=200, maxLineGap=15)

houghp = img.copy()
for x1, y1, x2, y2 in lines[0]:
    cv2.line(houghp, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow('HoughP', houghp)

cv2.waitKey(0)
cv2.destroyAllWindows()


# coding:utf-8

import cv2
import numpy as np


def hough_lines(canny, hough):
    lines = cv2.HoughLines(canny, 1, np.pi / 180, 160)  # 这里对最后一个参数使用了经验型的值
    print(lines.shape)
    for line in lines:
        rho = line[0][0]  # 第一个元素是距离rho
        theta = line[0][1]  # 第二个元素是角度theta
        if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
            # 该直线与第一行的交点
            pt1 = (int(rho / np.cos(theta)), 0)
            # 该直线与最后一行的焦点
            pt2 = (int((rho - hough.shape[0] * np.sin(theta)) / np.cos(theta)), hough.shape[0])
            # 绘制一条白线
            cv2.line(hough, pt1, pt2, (255))
        else:  # 水平直线
            # 该直线与第一列的交点
            pt1 = (0, int(rho / np.sin(theta)))
            # 该直线与最后一列的交点
            pt2 = (hough.shape[1], int((rho - hough.shape[1] * np.cos(theta)) / np.sin(theta)))
            # 绘制一条直线
            cv2.line(hough, pt1, pt2, (255), 1)
    return hough


shrink = 0.3

img = cv2.imread("C:/Users/qinq/Pictures/zto/8.jpg")

height, width, _ = img.shape

size = (int(width * shrink), int(height * shrink))

img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.GaussianBlur(gray, (3, 3), 0)
canny = cv2.Canny(edges, 100, 300)
cv2.imshow('canny', canny)

lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 60, minLineLength=50, maxLineGap=10)
print(lines.shape)
houghp = img.copy()
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(houghp, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow('HoughP', houghp)

cv2.waitKey()
cv2.destroyAllWindows()
