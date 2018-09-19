from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import sys
import random
import numpy as np

sys.path.append(os.getcwd())
from telephone import common


def deskew(img, szie=(8, 12)):
    affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR

    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, -2 * skew, -0.5 * 10 * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, szie, flags=affine_flags, borderValue=(255, 255, 255))
    return img


def genFontImage(font, char, image_size, center=(0, 0)):
    image = Image.new('1', image_size, color=255)
    draw = ImageDraw.Draw(image)
    draw.text(center, char, font=font, fill='#000000')
    return image


def run_(font_size=12, image_size=(10, 20), font_path='data/font/simfang.ttf', center=(0, 0)):
    font = ImageFont.truetype(font_path, font_size)
    hans = "0123456789"
    for han in hans[:10]:
        image = genFontImage(font, han, image_size, center)
        image.save("data/template/" + str(hans.index(han)) + '.png')


def fuse_bg(img):
    # img = cv2.filter2D(img, -1, np.array([[0, 1 / 4, 0], [1 / 4, 0, 1 / 4], [0, 1 / 4, 0]]))

    h, w = img.shape[:2]
    bg_path = ["data/fuse_bg/120514411790_1032.jpg", "data/fuse_bg/218428588707_1704.jpg",
               "data/fuse_bg/218428588707_1705.jpg", "data/fuse_bg/218583980807_2071.jpg"]

    bg_index = random.randint(0, 6)

    if bg_index >= len(bg_path):
        img[img > 200] = 200
    else:
        bg = cv2.imread(bg_path[bg_index])
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA)
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        bg_split = np.ones_like(bg)

        h_split = random.randint(1, h - 2)
        bg_split[0:h_split, :] = bg[(h - h_split):, :]
        bg_split[h_split:, :] = bg[0:(h - h_split), :]

        w_split = random.randint(1, w - 2)
        bg_split[:, 0:w_split] = bg[:, (w - w_split):]
        bg_split[:, w_split:] = bg[:, 0:(w - w_split)]

        img[img > 200] = bg_split[img > 200]

    img = cv2.blur(img, (3, 3))

    return img


def get_img(str="188", path='data/template', height=20, width=140):
    font_path = ['data/font/simfang.ttf', 'data/font/simhei.ttf', 'data/font/simkai.ttf']

    font_size = [16, 18, 14]
    center = [[(0, 2), (0, 2), (0, 2)],
              [(0, 2), (0, 2), (0, 2)],
              [(0, 2), (0, 2), (0, 2)]]
    random_path = np.random.randint(0, len(font_path))
    random_size = np.random.randint(0, len(font_size))
    if random_size == 2:
        run_(font_path=font_path[random_path],
             font_size=font_size[random_size],
             image_size=(8, 20),
             center=center[random_path][random_size])
    else:
        run_(font_path=font_path[random_path],
             font_size=font_size[random_size],
             image_size=(10, 20),
             center=center[random_path][random_size])

    images = None
    for _ in str:
        img = cv2.imread(os.path.join(path, _ + ".png"), 0)
        if images is None:
            images = img
        else:
            images = np.hstack((images, img))

    if random_size == 2:
        left = np.ones((images.shape[0], 8)) * 255
        right = np.ones((images.shape[0], 8)) * 255
        if random.randint(0, 3) == 0:
            images = np.hstack((images, left, right))
        elif random.randint(0, 3) == 1:
            images = np.hstack((left, images, right))
        elif random.randint(0, 3) == 2:
            images = np.hstack((left, right, images))

    # images = 255 - images

    img = cv2.resize(images, (width, height), interpolation=cv2.INTER_AREA)

    if random.randint(0, 1):
        img = deskew(img, (width, height))

    if random.randint(0, 2):
        degree = random.randint(-20, 20) / 10
        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
        img = cv2.warpAffine(img, matRotation, (width, height), borderValue=(255, 255, 255))

    offset_height = random.randint(-2, 2)
    M = np.float32([[1, 0, 0], [0, 1, offset_height]])
    img = cv2.warpAffine(img, M, (width, height), borderValue=(255, 255, 255))

    # noise
    img = common.SaltAndPepper(img, 0.2)  # 再添加10%的椒盐噪声
    img = fuse_bg(img)

    return img


if __name__ == '__main__':
    img = get_img(str="18852890100", run=True, font_path=None)
    cv2.imshow("image", img)
    cv2.imwrite("data/cut/_904.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
