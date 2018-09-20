# coding:utf-8
import sys
import os
import cv2
from lake.decorator import time_cost

import common
import numpy as np
import random
import math
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='[%Y-%m_%d %H:%M:%S]',
                    filename='log_module.log')


@time_cost
def get_image(file_name="./dataset/120353285332.jpg"):
    # 读取图片
    origin = cv2.imread(file_name)

    # 灰度化
    gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)

    # 二值化
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -10)

    img = binary
    # 中值滤波
    img = cv2.medianBlur(img, 3)

    # 形态学处理
    img = common.dilate_(img, ksize=(3, 3), iterations=3)

    img = common.morph_(img, ksize=(50, 50))

    # 轮廓提取
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 分割轮廓
    img = common.Find_BoundingRect(gray, contours)

    return img


def bar_code(binary, gray, min_ratio=2, min_area=80, max_area=30000):
    # 中值滤波
    img = cv2.medianBlur(binary, 3)
    # 形态学处理
    img = common.dilate_(img, ksize=(3, 3), iterations=4)
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    def max_filter(tour):
        area = cv2.contourArea(tour)
        if min_area <= area <= max_area:
            return True
        return False

    contours = filter(max_filter, contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[:min(25, int(0.2 * len(contours)))]
    angles = {}
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = np.int0(cv2.boxPoints(rect))

        if min_ratio <= common.distance(box[0], box[1]) / common.distance(box[1], box[2]):
            if box[1][0] - box[0][0] == 0:
                angle = 90
            else:
                angle = math.atan((box[1][1] - box[0][1]) / (box[1][0] - box[0][0])) * 180 / np.pi
        elif min_ratio <= common.distance(box[1], box[2]) / common.distance(box[0], box[1]):
            if box[1][0] - box[2][0] == 0:
                angle = 90
            else:
                angle = math.atan((box[1][1] - box[2][1]) / (box[1][0] - box[2][0])) * 180 / np.pi
        else:
            continue
        # cv2.drawContours(gray, [box], -1, (200, 200, 200), 3)
        key = (angle + 75) / 30
        if key < 0:
            key = key + 6
            angle = angle + 180
        key = int(key)
        value = angles.setdefault(key, [])
        value.append(angle)

    angles = sorted(angles.items(), key=lambda item: len(item[1]), reverse=True)

    key, value = angles[0]
    rot_angle = np.median(value)
    h, w = gray.shape[:2]
    if abs(rot_angle) > 45:
        side = max(w, h)
        center = (side / 2, side / 2)
        new_shape = (side, side)
    else:
        center = (w / 2, h / 2)
        new_shape = (w, h)
    M = cv2.getRotationMatrix2D(center, rot_angle, 1)
    gray = cv2.warpAffine(gray, M, new_shape)
    return gray


@time_cost
def get_image_of_path(path='./cut/dataset/'):
    dirs = os.listdir(path)
    for k, dirc in enumerate(dirs):
        origin = get_image(file_name="./cut/dataset/" + dirc)

        # 灰度变换
        gray = 255 - origin
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -10)

        gray = bar_code(binary, origin)

        # _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        h_left, h_right = common.border(binary, axis=1)
        w_left, w_right = common.border(binary, axis=0)
        gray = gray[h_left:h_right, w_left:w_right]
        # gray = common.transpose(gray, 0.8)
        #
        # if gray.shape[0] * gray.shape[1] < 1000 * 800:
        #     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        #     gray = common.roate_hough(edges, gray, angle_threshold=45)
        #     cv2.imwrite("./dataset_bin/" + dirc, gray)
        # else:
        #     gray_morph = common.binary_morph(gray, min_size=100)
        #     h_split = common.split_image(gray_morph, axis=1, split_gap=50)
        #
        #     for l, v in enumerate(h_split):
        #         temp = gray_morph[v[0]:v[1], :]
        #         h_left, h_right = common.border(temp, axis=1)
        #         w_left, w_right = common.border(temp, axis=0)
        #         temp = gray[v[0]:v[1], :][h_left:h_right, w_left:w_right]
        #         if temp.shape[0] * temp.shape[1] < 400 * 300:
        #             continue
        #
        #         temp = common.transpose(temp, 0.8)
        #
        #         edges = cv2.Canny(temp, 50, 150, apertureSize=3)
        #         temp = common.roate_hough(edges, temp, angle_threshold=30)
        #
        #         namespace = dirc.split('.')
        #         cv2.imwrite("./dataset_bin/" + namespace[0] + "_" + str(l) + "." + namespace[1], temp)
        cv2.imwrite("./cut/dataset_bin/" + dirc, gray)

        print("========" + dirc + "========")
        logging.info("========" + dirc + "========")


def select_largest_obj(img_bin, lab_val=255, fill_holes=False,
                       smooth_boundary=False, kernel_size=15, top=3):
    '''Select the largest object from a binary image and optionally
    fill holes inside it and smooth its boundary.
    Args:
        img_bin (2D array): 2D numpy array of binary image.
        lab_val ([int]): integer value used for the label of the largest
                object. Default is 255.
        fill_holes ([boolean]): whether fill the holes inside the largest
                object or not. Default is false.
        smooth_boundary ([boolean]): whether smooth the boundary of the
                largest object using morphological opening or not. Default
                is false.
        kernel_size ([int]): the size of the kernel used for morphological
                operation. Default is 15.
    Returns:
        a binary image as a mask for the largest object.
    '''
    n_labels, img_labeled, lab_stats, _ = \
        cv2.connectedComponentsWithStats(img_bin, connectivity=8,
                                         ltype=cv2.CV_32S)
    largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)

    # largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1
    # largest_mask[img_labeled == largest_obj_lab] = lab_val

    largest_obj_lab = np.argsort(-lab_stats[1:, 4]) + 1
    for lab in largest_obj_lab[:top]:
        largest_mask[img_labeled == lab] = lab_val

    if fill_holes:
        bkg_locs = np.where(img_labeled == 0)
        bkg_seed = (bkg_locs[0][0], bkg_locs[1][0])
        img_floodfill = largest_mask.copy()
        h_, w_ = largest_mask.shape
        mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
        cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed,
                      newVal=lab_val, loDiff=(20, 20, 20), upDiff=(20, 20, 20),
                      flags=8 | 1 << 8 | cv2.FLOODFILL_FIXED_RANGE)

        holes_mask = cv2.bitwise_not(img_floodfill)  # mask of the holes.
        largest_mask = largest_mask + holes_mask
    if smooth_boundary:
        kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        largest_mask = cv2.dilate(largest_mask, kernel_, iterations=5)

    return largest_mask


def test(dirc="120505227333"):
    dirc = dirc + ".jpg"
    origin = get_image(file_name="./cut/dataset/" + dirc)
    # 灰度变换
    gray = 255 - origin
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -10)

    gray = bar_code(binary, origin)

    # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    cv2.namedWindow("0", 0)
    cv2.imshow("0", binary)
    cv2.waitKey()
    h_left, h_right = common.border(binary, axis=1)
    w_left, w_right = common.border(binary, axis=0)
    gray = gray[h_left:h_right, w_left:w_right]

    # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -10)
    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # gray = common.roate_hough(edges, gray, angle_threshold=60)
    # cv2.imshow("gray", gray)
    #
    # w_left, w_right = common.border(gray, axis=0)
    # gray = gray[h_left:h_right, w_left:w_right]
    # gray = common.transpose(gray, 0.8)
    #
    # if gray.shape[0] * gray.shape[1] < 1000 * 800:
    #
    #     cv2.imwrite("./dataset_bin/" + dirc, gray)
    # else:
    #     gray_morph = common.binary_morph(gray, min_size=100)
    #     h_split = common.split_image(gray_morph, axis=1, split_gap=50)
    #
    #     for l, v in enumerate(h_split):
    #         temp = gray_morph[v[0]:v[1], :]
    #         h_left, h_right = common.border(temp, axis=1)
    #         w_left, w_right = common.border(temp, axis=0)
    #         if (h_right - h_left) * (w_right - w_left) < 400 * 300:
    #             continue
    #         temp = gray[v[0]:v[1], :][h_left:h_right, w_left:w_right]
    #         temp = common.transpose(temp, 0.8)
    #
    #         namespace = dirc.split('.')
    #         cv2.imwrite("./dataset_bin/" + namespace[0] + "_" + str(l) + "." + namespace[1], temp)

    cv2.imwrite("./cut/dataset_bin/" + dirc, gray)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print("========" + dirc + "========")
    logging.info("========" + dirc + "========")


if __name__ == "__main__":
    get_image_of_path()
    # test("217706725913")
    # os.popen("robocopy  ./dataset_bin ../text-detection-ctpn/data/demo")
    # os.popen("cd ../text-detection-ctpn")
    # os.popen("python ../text-detection-ctpn/ctpn/demo_pb.py")
