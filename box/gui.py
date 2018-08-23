import sys
import os
import cv2
import math
import numpy as np

sys.path.append(os.getcwd())
import box.cut_telephone as cut
from telephone import common

global img, name
global point1, point2, angle
angle = 0


def on_mouse(event, x, y, flags, param):
    global point1, point2
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 0, 0), 2)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        cv2.rectangle(img2, point1, (x, y), (0, 0, 0), 2)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 0), 2)
        cv2.imshow('image', img2)


def rotate_(angle):
    center = [0.0, 0.0]
    half = [0.0, 0.0]
    img2 = img.copy()

    center[0] = (point1[0] + point2[0]) / 2
    center[1] = (point1[1] + point2[1]) / 2

    half[0] = abs(point1[0] - point2[0]) / 2
    half[1] = abs(point1[1] - point2[1]) / 2

    pt1 = (center[0] - half[0], center[1] - half[1])
    pt2 = (center[0] + half[0], center[1] - half[1])
    pt3 = (center[0] - half[0], center[1] + half[1])
    pt4 = (center[0] + half[0], center[1] + half[1])

    cut.drawRect(img2, pt1, pt2, pt3, pt4, center, angle, color=0, lineWidth=2)
    cv2.imshow('image', img2)


def render(img):
    rotate_(angle)
    # cv2.rectangle(img, point1, point2, (0, 0, 0), 2)
    # cv2.imshow('image', img)


def main(path='data/min_img', file_name='1.jpg', prefix='data/cut/_'):
    global img, name
    global point1, point2, angle

    name = os.path.join(path, file_name)
    img = cv2.imread(name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', img)
    while True:
        key = cv2.waitKey(0)
        # up
        if key == 2490368:
            point1 = (point1[0], point1[1] - 1)
            point2 = (point2[0], point2[1] - 1)
            render(img.copy())
        # down
        elif key == 2621440:
            point1 = (point1[0], point1[1] + 1)
            point2 = (point2[0], point2[1] + 1)
            render(img.copy())
        # left
        elif key == 2424832:
            point1 = (point1[0] - 1, point1[1])
            point2 = (point2[0] - 1, point2[1])
            render(img.copy())
        # right
        elif key == 2555904:
            point1 = (point1[0] + 1, point1[1])
            point2 = (point2[0] + 1, point2[1])
            render(img.copy())
        # rotate_
        elif key == 109:
            angle = (angle + 1) % 180
            rotate_(angle)
            # rotate_
        elif key == 110:
            angle = (angle - 1) % 180
            rotate_(angle)

        elif key == 27:
            row = int((point1[0] + point2[0]) / 2)
            col = int((point1[1] + point2[1]) / 2)
            width = abs(point1[0] - point2[0])
            height = abs(point1[1] - point2[1])

            img, cut_img = cut.rotate_(img, (row, col), (width, height), angle, draw=False, display=False)
            cut_img = common.resize_(255 - cut_img, width=140, height=20)
            cv2.imwrite(prefix + file_name, cut_img)
            cv2.destroyAllWindows()
            return
        # + height(W)
        elif key == 115:
            point2 = (point2[0], point2[1] + 1)
            render(img.copy())

        # - height(S)
        elif key == 119:
            point2 = (point2[0], point2[1] - 1)
            render(img.copy())

        # + width(A)
        elif key == 100:
            point2 = (point2[0] + 1, point2[1])
            render(img.copy())

        # - width(D)
        elif key == 97:
            point2 = (point2[0] - 1, point2[1])
            render(img.copy())


# 黑:0
# 白:255
if __name__ == '__main__':
    main(file_name="1.jpg", prefix='data/min_img/_')
