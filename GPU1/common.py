# coding:utf-8
import cv2
import numpy as np


def resize_(img, shrink=0.3):
    height, width = img.shape[:2]
    size = (int(width * shrink), int(height * shrink))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


# sqrt
def hist_impore_(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) / (cdf_m.max() - cdf_m.min())
    cdf_m = cdf_m * cdf_m * cdf_m * cdf_m * cdf_m
    cdf_m = (cdf_m - cdf_m.min()) / (cdf_m.max() - cdf_m.min())

    cdf_m = cdf_m * 255
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    equ = cdf[img]
    return equ


def equalizeHist_(img, improve=0):
    for _ in range(img.shape[2]):
        if improve == 0:
            img[:, :, _] = cv2.equalizeHist(img[:, :, _])
        elif improve == 1:
            img[:, :, _] = hist_impore_(img[:, :, _])

    return img


def bgr2gray_(img):
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def blur_(img, ksize=(5, 5)):
    return cv2.blur(img, ksize)


# 100 200
def binary_(img, thresh=100):
    ret, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return binary


def floodFill_(img):
    h, w = img.shape[:2]
    mask = np.zeros([h + 2, w + 2], np.uint8)
    return cv2.floodFill(img, mask=mask, seedPoint=(30, 30), newVal=(0, 255, 255),
                         loDiff=(100, 100, 100), upDiff=(50, 50, 50),
                         flags=cv2.FLOODFILL_FIXED_RANGE)


def canny_(img, ksize=(3, 3), threshold1=100, threshold2=300):
    edges = cv2.GaussianBlur(img, ksize, 0)
    canny = cv2.Canny(edges, threshold1, threshold2)
    return canny


def harris_(img, gray, thread=0.01):
    # 角检测
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > thread * dst.max()] = [0, 0, 255]
    return img


def sobel_(img):
    # sobel 水平方向边缘检测
    # edges = cv2.Sobel(img, cv2.CV_16S, 1, 0)

    # sobel 竖直方向边缘检测
    # edges = cv2.Sobel(img, cv2.CV_16S, 0, 1)

    # sobel全方向边缘检测
    edges = cv2.Sobel(img, cv2.CV_16S, 1, 1)
    # 浮点型转成uint8型
    edges = cv2.convertScaleAbs(edges)
    return edges


def erode_(img, ksize=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    # 腐蚀
    img = cv2.erode(img, kernel)
    return img


def dilate_(img, ksize=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    # 膨胀
    img = cv2.dilate(img, kernel)
    return img


def morph_(img, ksize=(20, 20)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    # 闭运算
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # 开运算
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img


def findContours_(img, origin, draw=False):
    cloneImg, contours, heriachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if draw:
        cv2.drawContours(origin, contours, -1, (0, 0, 255), 2)
        cv2.imshow("findContours_", origin)
    return contours


def boundingRect_(img, contours, draw=False, min_area=100 * 50):
    x = y = w = h = 0
    for tours in contours:
        x_, y_, w_, h_ = cv2.boundingRect(tours)
        if w_ * h_ >= min_area:
            min_area = w_ * h_
            x, y, w, h = x_, y_, w_, h_
    if w * h == 0:
        return None
    if draw:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255))
        cv2.imshow("boundingRect_", img)

    region = img[y:y + h, x:x + w, :]
    return region


def minAreaRect_(img, contours, draw=False):
    tours = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    x, y, w, h = cv2.boundingRect(tours)
    region = img[y:y + h, x:x + w, :]

    # 中心（x，y），（宽度，高度），旋转角度）
    rect = cv2.minAreaRect(tours)
    box = np.int0(cv2.boxPoints(rect))
    if draw:
        cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
        cv2.imshow("minAreaRect_", img)

    # 仿射变换、透视变换
    # 负数，顺时针旋转
    center = rect[0]
    area = np.int0(rect[1])
    angle = rect[2]
    if 0 < abs(angle) <= 45:
        angle = angle
    # 正数，逆时针旋转
    elif 45 < abs(angle) < 90:
        angle = 90 - abs(angle)

    roateM = cv2.getRotationMatrix2D(center=(center[0] - x, center[1] - y), angle=angle, scale=1)
    # 仿射变换
    roi = cv2.warpAffine(region, roateM, (w, h))

    return roi


def grabCut_(img, newmask):
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[newmask == 0] = 0
    mask[newmask == 255] = 1
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask[:, :, np.newaxis]
    return img, mask


def template(img, temp):
    if len(img.shape) == 3:
        img_ = bgr2gray_(img)
    else:
        img_ = img
    if len(temp.shape) == 3:
        temp_ = bgr2gray_(temp)
    else:
        temp_ = temp
    h, w = temp_.shape

    # cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED,cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED
    method = cv2.TM_SQDIFF_NORMED
    res = cv2.matchTemplate(img_, temp_, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    # 设定阈值
    threshold = 0.1
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        # min
        threshold = 1 - threshold
        loc = np.where(res <= threshold)
    else:
        # max
        loc = np.where(res >= threshold)
    for pt in zip(*loc[:: -1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), 249, 2)
    return img


def hough_lines_p_(edges, img):
    """
    :param edges:边缘
    :param img: 图片
    :return: img
    """
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    for line in lines:
        print(line)
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img


def hough_lines_(edges, img):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    for line in lines:
        rho = line[0][0]  # 第一个元素是距离rho
        theta = line[0][1]  # 第二个元素是角度theta
        if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
            # 该直线与第一行的交点
            pt1 = (int(rho / np.cos(theta)), 0)
            # 该直线与最后一行的焦点
            pt2 = (int((rho - img.shape[0] * np.sin(theta)) / np.cos(theta)), img.shape[0])
            # 绘制一条白线
            cv2.line(img, pt1, pt2, (255))
        else:  # 水平直线
            # 该直线与第一列的交点
            pt1 = (0, int(rho / np.sin(theta)))
            # 该直线与最后一列的交点
            pt2 = (img.shape[1], int((rho - img.shape[1] * np.cos(theta)) / np.sin(theta)))
            # 绘制一条直线
            cv2.line(img, pt1, pt2, (255), 1)
    return img

# randon
# DFT