# coding:utf-8
import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.getcwd())
from telephone import common
from telephone import config
from telephone import hand_write

config = config.config(100)


def read_(file_name, shrink=0.5):
    # 颜色空间
    origin = cv2.imread(file_name)
    origin = common.resize_(origin, shrink=shrink)
    # origin = common.equalizeHist_(origin)
    # config.binary_threshold = 200
    return origin


def roi_(origin):
    img = common.binary_(origin, thresh=config.binary_threshold)
    img = common.erode_(img)
    img = common.morph_(img)

    img = common.canny_(img)
    contours = common.findContours_(img, origin)

    # FFT 矫正
    roi = common.minAreaRect_(origin, contours)

    return roi


def cut_(origin):
    img = common.bgr2gray_(origin)
    img = common.binary_(img, thresh=config.binary_threshold)

    cut, mask = common.grabCut_(origin, newmask=img)

    return cut


def chosen(origin):
    # randon hough
    img = common.binary_(origin, thresh=config.binary_threshold)
    img = common.erode_(img)
    img = common.morph_(img)

    img = common.canny_(img)
    origin = common.hough_lines_p_(img, origin)

    contours = common.findContours_(img, origin)
    roi = common.boundingRect_(origin, contours)

    return roi


def run():
    path = "data/img/"
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        img = read_(file_path)
        roi = roi_(img)
        cv2.imshow(file, roi)
    cv2.waitKey()
    cv2.destroyAllWindows()


def genFontImage(font, char, image_size):
    image = Image.new('1', image_size, color=0)
    draw = ImageDraw.Draw(image)
    draw.text((0, -5), char, font=font, fill='#FFFFFF')
    image.save('temp.jpg')




# 11,12,13
def main_(file_path="../data/img/1_1.jpg"):
    img = read_(file_path, shrink=1)
    img = cut_(img)
    img = roi_(img)

    if img.shape[0] * img.shape[1] > 400 * 400:
        img = chosen(img)
    img = 255 - img
    img = common.bgr2gray_(img)
    # # img = common.canny_(img)
    # # img = common.Remove_holes(img)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -10)
    # img = cv2.bilateralFilter(img, 5, 100, 10)
    # img = common.Removing_small_connected_domain(img, 5)
    cv2.imshow("chosen", img)

    genFontImage(ImageFont.truetype('../data/font/msyhbd.ttf', 22), '2', (14, 20))
    temp = cv2.imread("temp.jpg")
    # temp = common.canny_(temp)
    # temp = common.Remove_holes(temp)
    cv2.imshow("temp", temp)

    img = common.template(img, temp, threshold=1)
    cv2.imshow("match", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main_()
