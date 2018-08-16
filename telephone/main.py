# coding:utf-8
import sys
import os
import cv2

sys.path.append(os.getcwd())
from telephone import common
from telephone import hand_write
from telephone import config

hand_write.run_()
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
    path = "../data/img/"
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        img = read_(file_path)
        roi = roi_(img)
        cv2.imshow(file, roi)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 11,12,13
def main_(file_path="../data/img/11.jpg"):
    img = read_(file_path, shrink=1)
    img = cut_(img)
    cv2.imwrite("../data/result/1_cut.jpg", img)
    img = roi_(img)
    cv2.imwrite("../data/result/2_roi.jpg", img)
    if img.shape[0] * img.shape[1] > 400 * 400:
        img = chosen(img)
    cv2.imwrite("../data/result/3_chosen.jpg", img)
    img = common.canny_(img)

    temp = hand_write.get_img("872")
    temp = common.canny_(temp)
    cv2.imwrite("../data/result/4_temp.jpg", temp)

    img = common.template(img, temp)
    cv2.imwrite("../data/result/5_match.jpg", img)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main_()
