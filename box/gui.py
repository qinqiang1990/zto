import cv2
import math
import numpy as np

global img, name
global point1, point2, angle
angle = 0


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


def drawRect(img, pt1, pt2, pt3, pt4, center=(0, 0), angle=0, color=(0, 255, 0), lineWidth=2):
    img2 = img.copy()

    pt1 = coordinate(angle, center, pt1, img2.shape[0])
    pt2 = coordinate(angle, center, pt2, img2.shape[0])
    pt3 = coordinate(angle, center, pt3, img2.shape[0])
    pt4 = coordinate(angle, center, pt4, img2.shape[0])

    cv2.line(img2, pt1, pt2, color, lineWidth)
    cv2.line(img2, pt1, pt3, color, lineWidth)
    cv2.line(img2, pt2, pt4, color, lineWidth)
    cv2.line(img2, pt3, pt4, color, lineWidth)

    cv2.imshow('image', img2)


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
    print(angle)
    center = [0.0, 0.0]
    half = [0.0, 0.0]

    center[0] = (point1[0] + point2[0]) / 2
    center[1] = (point1[1] + point2[1]) / 2

    half[0] = abs(point1[0] - point2[0]) / 2
    half[1] = abs(point1[1] - point2[1]) / 2

    pt1 = (center[0] - half[0], center[1] - half[1])
    pt2 = (center[0] + half[0], center[1] - half[1])
    pt3 = (center[0] - half[0], center[1] + half[1])
    pt4 = (center[0] + half[0], center[1] + half[1])

    drawRect(img, pt1, pt2, pt3, pt4, center, angle)


def render(img):
    cv2.rectangle(img, point1, point2, (0, 0, 0), 2)
    cv2.imshow('image', img)


def main():
    global img, name
    global point1, point2, angle

    name = 'cut.jpg'
    img = cv2.imread(name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    cv2.createTrackbar('angle', 'image', 0, 180, rotate_)
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
        elif key == 110:
            angle = (angle + 1) % 180
            rotate_(angle)
            # rotate_
        elif key == 109:
            angle = (angle - 1) % 180
            rotate_(angle)

        elif key == 27:
            min_x = min(point1[0], point2[0])
            min_y = min(point1[1], point2[1])
            width = abs(point1[0] - point2[0])
            height = abs(point1[1] - point2[1])
            cut_img = 255 - img[min_y:min_y + height, min_x:min_x + width]
            cv2.imwrite("1_" + name, cut_img)
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
    main()
