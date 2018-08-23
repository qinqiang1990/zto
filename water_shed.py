import numpy as np
import cv2
from matplotlib import pyplot as plt
import FFT
import telephone.common as common


# 去除孔洞
def Remove_holes(thresh, AreaLimit):
    label = np.zeros_like(thresh)
    label[thresh != 0] = 3

    # 4 领域
    NeihborPos = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    h, w = thresh.shape[:2]
    for i in range(h):
        for j in range(w):

            if label[i, j] == 0:
                GrowBuffer = []
                GrowBuffer.append((i, j))
                label[i, j] = 1
                k = 0
                while k < len(GrowBuffer):
                    cur = GrowBuffer[k]
                    for pos in NeihborPos:
                        cur_h = cur[0] + pos[0]
                        cur_w = cur[1] + pos[1]
                        if 0 <= cur_h < h and 0 <= cur_w < w:
                            if label[cur_h, cur_w] == 0:
                                GrowBuffer.append((cur_h, cur_w))
                                label[cur_h, cur_w] = 1
                    k = k + 1
                # 判断结果（是否超出限定的大小），1为未超出，2为超出
                if len(GrowBuffer) > AreaLimit:
                    check_result = 2
                else:
                    check_result = 1

                for pos in GrowBuffer:
                    # 标记不合格的像素点，像素值为2
                    label[pos[0], pos[1]] += check_result

    check_mode = 255
    img = thresh.copy()
    img[label == 2] = check_mode

    return img


# 去除小连通域
def Removing_small_connected_domain(thresh, AreaLimit):
    label = np.zeros_like(thresh)
    label[thresh == 0] = 3

    # 8 领域
    NeihborPos = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    h, w = thresh.shape[:2]
    for i in range(h):
        for j in range(w):

            if label[i, j] == 0:
                GrowBuffer = []
                GrowBuffer.append((i, j))
                label[i, j] = 1
                k = 0
                while k < len(GrowBuffer):
                    cur = GrowBuffer[k]
                    for pos in NeihborPos:
                        cur_h = cur[0] + pos[0]
                        cur_w = cur[1] + pos[1]
                        if 0 <= cur_h < h and 0 <= cur_w < w:
                            if label[cur_h, cur_w] == 0:
                                GrowBuffer.append((cur_h, cur_w))
                                label[cur_h, cur_w] = 1
                    k = k + 1
                # 判断结果（是否超出限定的大小），1为未超出，2为超出
                if len(GrowBuffer) > AreaLimit:
                    check_result = 2
                else:
                    check_result = 1

                for pos in GrowBuffer:
                    # 标记不合格的像素点，像素值为2
                    label[pos[0], pos[1]] += check_result

    check_mode = 0
    img = thresh.copy()
    img[label == 2] = check_mode

    return img


def border(area, ratio=0.85, gap=100, axis=1):
    '''
    :param area:
    :param ratio:
    :param gap:
    :param axis: 1:height;0:width
    :return:
    '''
    h = np.sum(area, axis=axis)
    h = h > ratio * np.mean(h)
    h = [_ for _ in range(len(h)) if h[_] == 1]
    h = [h[i] for i in range(len(h)) if i == 0 or i == len(h) - 1 or h[i] > h[i - 1] + gap]
    print(h)
    max_border = 0
    for i in range(1, len(h)):
        if h[i] - h[i - 1] > max_border:
            max_border = h[i] - h[i - 1]
            h_left = h[i - 1]
            h_right = h[i]
    return h_left, h_right


img = cv2.imread('roi.jpg')
img = common.equalizeHist_(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -10)

area = Removing_small_connected_domain(thresh, 100)

cv2.imwrite("roi_.jpg", area)
area = common.dilate_(area, ksize=(5, 5))

h_left, h_right = border(area, axis=1)
w_left, w_right = border(area, axis=0)

cv2.imwrite("roi_1.jpg", gray[h_left:h_right, w_left:w_right])
