# coding:utf-8
import cv2
import numpy as np
import random


# https://blog.csdn.net/u012936765/article/details/53200918
# 定义添加高斯噪声的函数
def addGaussianNoise(image, loc=30, scale=30):
    PixcelMin = 0
    PixcelMax = 255
    h, w = image.shape
    G_Noiseimg = np.random.normal(loc=loc, scale=scale, size=h * w).reshape(h, w)
    G_Noiseimg = G_Noiseimg + image

    G_Noiseimg[G_Noiseimg > PixcelMax] = PixcelMax
    G_Noiseimg[G_Noiseimg < PixcelMin] = PixcelMin
    return G_Noiseimg.astype(np.uint8)


# 定义添加椒盐噪声的函数
def SaltAndPepper(src, percetage):
    SP_NoiseImg = src
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)

        SP_NoiseImg[randX, randY] = 0
        # if random.randint(0, 1) == 0:
        #     SP_NoiseImg[randX, randY] = 0
        # else:
        #     SP_NoiseImg[randX, randY] = 255
    return SP_NoiseImg


def resize_(img, shrink=0.3, width=None, height=None):
    height_, width_ = img.shape[:2]
    if shrink is not None:
        size = (int(width_ * shrink), int(height_ * shrink))
    if width is not None and height is not None:
        size = (width, height)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def equalizeHist_(img):
    for _ in range(img.shape[2]):
        img[:, :, _] = cv2.equalizeHist(img[:, :, _])
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
    # binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -10)
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


def template(img, temp, threshold=None):
    if len(img.shape) == 3:
        img_ = bgr2gray_(img)
    else:
        img_ = img
    if len(temp.shape) == 3:
        temp_ = bgr2gray_(temp)
    else:
        temp_ = temp
    h, w = temp_.shape

    # cv2.TM_CCORR, cv2.TM_CCORR_NORMED (max)
    # cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED (max)
    # cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED (min)
    method = cv2.TM_CCORR_NORMED
    res = cv2.matchTemplate(img_, temp_, method)
    cv2.normalize(res, res, 0, 1, cv2.NORM_MINMAX)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    if threshold is not None:
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


def filter_(img, kernel=None):
    # kernel = np.array([[0, -2, 0], [-2, 9, -2], [0, -2, 0]])
    return cv2.filter2D(img, -1, kernel)


# 去除孔洞
def Remove_holes(thresh, AreaLimit=100):
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


# method: mean | otsu
def bin_(img, method="mean", rate=0.9, bais=10):
    hh, ww = img.shape

    height = math.sqrt(hh / ww * (hh + ww))
    width = ww / hh * height

    mat = np.zeros((hh, ww))
    for i in range(hh):
        for j in range(ww):

            h1 = i - int(height / 2)
            h2 = i + int(height / 2)
            w1 = j - int(width / 2)
            w2 = j + int(width / 2)
            if h1 < 0:
                h1 = 0
            if w1 < 0:
                w1 = 0
            if h2 >= hh:
                h2 = hh - 1
            if w2 >= ww:
                w2 = ww - 1

            if method == "mean":
                mat[i, j] = np.mean(img[h1:h2, w1:w2]) + bais
            elif method == "otsu":
                mat[i, j], _ = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    mat[img < mat * rate] = 0
    mat[mat != 0] = 255

    return mat

# randon
