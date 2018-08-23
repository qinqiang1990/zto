import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

VERTICAL = 0
HORIZONTAL = 1


def fourier(gray, method=None):
    '''
    :param gray:
    :param method:VERTICAL,HORIZONTAL
    :return:
    '''
    if method == None:
        return None
    # 2、图像延扩
    h, w = gray.shape
    new_h = cv.getOptimalDFTSize(h)
    new_w = cv.getOptimalDFTSize(w)
    right = new_w - w
    bottom = new_h - h
    nimg = cv.copyMakeBorder(gray, 0, bottom, 0, right, borderType=cv.BORDER_CONSTANT, value=0)

    # 3、执行傅里叶变换，并过得频域图像
    f = np.fft.fft2(nimg)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift))

    # 二值化
    magnitude_uint = magnitude.astype(np.uint8)
    ret, thresh = cv.threshold(magnitude_uint, 11, 255, cv.THRESH_BINARY)
    cv.imshow('thresh', thresh)
    # 霍夫直线变换
    lines = cv.HoughLinesP(thresh, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=100)

    # 创建一个新图像，标注直线
    lineimg = np.ones(nimg.shape, dtype=np.uint8)
    lineimg = lineimg * 255

    theta = 0.0
    number = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(lineimg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if x2 - x1 == 0:
            continue
        else:
            theta_ = math.atan((y2 - y1) / (x2 - x1))
        if method == HORIZONTAL and np.pi / 180 < abs(theta_) < np.pi / 4:
            theta_ = theta_ * (180 / np.pi)
            theta += theta_ if (theta_ > 0) else theta_ + 180
            number += 1

        if method == VERTICAL and np.pi / 4 < abs(theta_) < np.pi / 2:
            theta_ = theta_ * (180 / np.pi)
            theta += theta_ if (theta_ > 0) else theta_ + 180
            number += 1

    if method == HORIZONTAL:
        angle = theta / number - 180
        print(angle)
        center = (w / 2, h / 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv.warpAffine(gray, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    if method == VERTICAL:
        angle = theta / number
        print(angle)
        M = np.float32([[1, math.cos(angle * math.pi / 180), 0], [0, 1, 0]])
        rotated = cv.warpAffine(gray, M, (w, h))

    cv.imshow('gray', gray)
    cv.imshow('line image', lineimg)
    cv.imshow('rotated', rotated)
    cv.imwrite("rotated.jpg", rotated)
    return rotated


if __name__ == '__main__':
    # 1、读取文件，灰度化
    file_name = 'data/min_img/15.jpg'
    img = cv.imread(file_name)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray = fourier(gray, method=HORIZONTAL)
    # gray = fourier(gray, method=VERTICAL)
    cv.imwrite(file_name, gray)

    cv.waitKey(0)
    cv.destroyAllWindows()
