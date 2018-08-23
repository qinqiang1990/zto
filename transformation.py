import numpy as np
import cv2
import sys
import math


def resize_(img, width, height):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)


def move_(img, affineShrink=np.array([[0.5, 0, 0], [0, 0.5, 0]], np.float32)):
    # 仿射变换矩阵，将图像缩小两倍
    affineShrink = np.array([[0.5, 0, 0], [0, 0.5, 0]], np.float32)
    shrinkTwoTimes = cv2.warpAffine(image, affineShrink, (int(cols / 2), int(rows / 2)), borderValue=125)

    # 先缩小2倍再平移
    affineShrinkTranslation = np.array([[0.5, 0, cols / 4], [0, 0.5, rows / 4]], np.float32)
    shrinkTwoTimesTranslation = cv2.warpAffine(image, affineShrinkTranslation, (cols, rows), borderValue=125)


def Rotation(img, angle=30):
    # 在shrinkTwoTimesTranslation的基础上 绕图像的中心旋转
    affineShrinkTranslationRotation = cv2.getRotationMatrix2D((int(cols / 2), int(rows / 2)), angle, 1)
    ShrinkTranslationRotation = cv2.warpAffine(img, affineShrinkTranslationRotation, (cols, rows),
                                               borderValue=125)


def Affine_(img):
    # 输入图像中的三个点
    rows, cols = img.shape[:2]
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

    M = cv2.getAffineTransform(pts1, pts2)

    return cv2.warpAffine(img, M, (cols, rows))


def Perspective_(img):
    # 输入图像上需要4个点
    rows, cols = img.shape[:2]
    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    return cv2.warpPerspective(img, M, (300, 300))


# 读取原始的图像
image = cv2.imread('1.jpg', 1)

# 原图的高、宽 以及通道数
rows, cols, channel = image.shape

cv2.imshow('image', Perspective_(image))
cv2.waitKey(0)
cv2.destroyAllWindows()
