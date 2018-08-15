from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import numpy as np


def genFontImage(font, char, image_size):
    image = Image.new('1', image_size, color=255)
    draw = ImageDraw.Draw(image)
    draw.text((0, -3), char, font=font, fill='#000000')
    return image


def run_():
    font_size = 25
    image_size = (12, 18)

    font = ImageFont.truetype('./times.ttf', font_size)
    hans = "0123456789"

    for han in hans[:10]:
        image = genFontImage(font, han, image_size)
        image.save("./data/" + str(hans.index(han)) + '.png')


def get_img(str="187", path='./data', run=False):
    if run:
        run_()
    images = None
    for _ in str:
        img = cv2.imread(os.path.join(path, _ + ".png"), 0)
        if images is None:
            images = img
        else:
            images = np.hstack((images, img))
    return images


if __name__ == '__main__':
    get_img()
