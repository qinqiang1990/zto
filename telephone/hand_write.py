from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import sys
import numpy as np

sys.path.append(os.getcwd())
from telephone import common


def genFontImage(font, char, image_size):
    image = Image.new('1', image_size, color=255)
    draw = ImageDraw.Draw(image)
    draw.text((0, -4), char, font=font, fill='#000000')
    return image


def run_():
    font_size = 20
    image_size = (12, 20)

    # font = ImageFont.truetype('data/font/msyhbd.ttf', font_size)
    font = ImageFont.truetype('data/font/times.ttf', font_size)
    hans = "0123456789"

    for han in hans[:10]:
        image = genFontImage(font, han, image_size)
        image.save("data/template/" + str(hans.index(han)) + '.png')


def get_img(str="188", path='data/template', run=False):
    if run:
        run_()
    images = None
    for _ in str:
        img = cv2.imread(os.path.join(path, _ + ".png"), 0)
        if images is None:
            images = img
        else:
            images = np.hstack((images, img))
    return 255 - images


if __name__ == '__main__':
    img = get_img(str="18852890100", run=True)
    img = common.erode_(img, ksize=(1, 1))
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
