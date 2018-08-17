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
    draw.text((0, 0), char, font=font, fill='#000000')
    return image


def run_(font_size=16, image_size=(10, 20), font_path='data/font/simfang.ttf'):
    font = ImageFont.truetype(font_path, font_size)
    hans = "0123456789"
    for han in hans[:10]:
        image = genFontImage(font, han, image_size)
        image.save("data/template/" + str(hans.index(han)) + '.png')


def get_img(str="188", path='data/template', run=False, font_path=None, height=20, width=140):
    if run:
        fonts = ['data/font/MSYHBD.TTC', 'data/font/msyhbd.ttf', 'data/font/MSYH.TTC']
        if font_path is None:
            font_path = fonts[np.random.randint(0, len(fonts) - 1)]
        run_(font_path=font_path)
    images = None
    for _ in str:
        img = cv2.imread(os.path.join(path, _ + ".png"), 0)
        if images is None:
            images = img
        else:
            images = np.hstack((images, img))
    images = 255 - images

    left = np.zeros((images.shape[0], 5))
    right = np.zeros((images.shape[0], 5))
    images = np.hstack((left, images, right))

    img = cv2.resize(images, (width, height), interpolation=cv2.INTER_AREA)

    # noise
    img = common.SaltAndPepper(img, 0.15)  # 再添加10%的椒盐噪声
    img = common.addGaussianNoise(img, 30, 30)  # 高斯噪声

    return img


if __name__ == '__main__':
    img = get_img(str="18852890100", run=True, font_path=None)
    cv2.imshow("image", img)
    cv2.imwrite("data/cut/_904.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
