import telephone.common as common
import matplotlib.pyplot as plt
import cv2
import os
import math
import numpy as np


# %matplotlib inline

# P(x1,y1)，绕坐标点Q(x2,y2)旋转θ角度后,新的坐标设为(x, y)的计算公式
# x= (x1 - x2)*cos(θ) - (y1 - y2)*sin(θ) + x2 ;
# y= (x1 - x2)*sin(θ) + (y1 - y2)*cos(θ) + y2 ;
def coordinate(angle, center, point, row):
    x1, y1 = point
    x2, y2 = center
    angle = math.radians(angle)
    x1 = x1
    y1 = row - y1
    x2 = x2
    y2 = row - y2
    x = (x1 - x2) * math.cos(angle) - (y1 - y2) * math.sin(angle) + x2
    y = (x1 - x2) * math.sin(angle) + (y1 - y2) * math.cos(angle) + y2
    x = np.int0(x)
    y = np.int0(row - y)
    return (x, y)


def drawRect(img, pt1, pt2, pt3, pt4, center=(0, 0), angle=0, color=160, lineWidth=5):
    pt1 = coordinate(angle, center, pt1, img.shape[0])
    pt2 = coordinate(angle, center, pt2, img.shape[0])
    pt3 = coordinate(angle, center, pt3, img.shape[0])
    pt4 = coordinate(angle, center, pt4, img.shape[0])

    cv2.line(img, pt1, pt2, color, lineWidth)
    cv2.line(img, pt1, pt3, color, lineWidth)
    cv2.line(img, pt2, pt4, color, lineWidth)
    cv2.line(img, pt3, pt4, color, lineWidth)


# center（row,col）
# shape（width,height）
def rotate_(img, center, shape, angle=0, draw=False, display=True):
    half = (np.int(shape[0] / 2), np.int(shape[1] / 2))
    pt1 = (center[0] - half[0], center[1] - half[1])
    pt2 = (center[0] + half[0], center[1] - half[1])
    pt3 = (center[0] - half[0], center[1] + half[1])
    pt4 = (center[0] + half[0], center[1] + half[1])

    # 画出矩形框
    if draw:
        drawRect(img, pt1, pt2, pt3, pt4, center, angle)

    # 按angle角度旋转图像
    height, width = img.shape
    angle = -angle
    rotateMat = cv2.getRotationMatrix2D(center, angle, 1)
    imgRotation = cv2.warpAffine(img, rotateMat, (width, height), 250)

    # 截取矩形框
    cut = imgRotation[pt1[1]:pt4[1], pt1[0]:pt4[0]]

    if display:
        plt.figure(figsize=(10, 10))

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        ax = plt.gca()  # 获取到当前坐标轴信息
        ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面

        plt.subplot(1, 2, 2)
        plt.imshow(cut)
        ax = plt.gca()  # 获取到当前坐标轴信息
        ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
        plt.show()

    #     ret, cut = cv2.threshold(cut, 100, 255, cv2.THRESH_BINARY)

    return img, cut


if __name__ == '__main__':
    # 0黑色，255白色
    # 11,12,13
    file_name = "../data/img/2.jpg"
    img = cv2.imread(file_name)
    img = common.bgr2gray_(img)
    img = common.binary_(img)

    img, cut = rotate_(img, (1430, 895), (280, 30), 30, draw=False, display=True)

    print(cut)
