# coding=utf-8
"""
身份证文字+数字生成类

"""
import sys
import os

import numpy as np
import freetype
import copy
import random
import cv2

sys.path.append(os.getcwd())
from telephone import common


class put_chinese_text(object):
    def __init__(self, ttf):
        self.ttf = ttf
        self._face = freetype.Face(ttf)

    def draw_text(self, image, pos, text, text_size, text_color):
        '''
        draw chinese(or not) text with ttf
        :param image:     image(numpy.ndarray) to draw text
        :param pos:       where to draw text
        :param text:      the context, for chinese should be unicode type
        :param text_size: text size
        :param text_color:text color
        :return:          image
        '''
        self._face.set_char_size(text_size * 64)
        metrics = self._face.size
        ascender = metrics.ascender / 64.0
        # descender = metrics.descender/64.0
        # height = metrics.height/64.0
        # linegap = height - ascender + descender
        ypos = int(ascender)

        if not isinstance(text, str):
            text = text.decode('utf-8')
        img = self.draw_string(image, pos[0], pos[1] + ypos, text, text_color)
        return img

    def draw_string(self, img, x_pos, y_pos, text, color, gap=1):
        '''
        draw string
        :param x_pos: text x-postion on img
        :param y_pos: text y-postion on img
        :param text:  text (unicode)
        :param color: text color
        :return:      image
        '''

        if self.ttf == "data/font/times.ttf":
            gap = gap * 1.2
        elif self.ttf == "data/font/msyhbd.ttf":
            gap = gap * 1
        else:
            gap = gap
        prev_char = 0
        pen = freetype.Vector()
        pen.x = x_pos << 6  # div 64
        pen.y = y_pos << 6

        hscale = 1.0
        matrix = freetype.Matrix(int(hscale) * 0x10000, int(0.2 * 0x10000), int(0.0 * 0x10000), int(1.1 * 0x10000))
        cur_pen = freetype.Vector()
        pen_translate = freetype.Vector()

        image = copy.deepcopy(img)
        for cur_char in text:
            self._face.set_transform(matrix, pen_translate)

            self._face.load_char(cur_char)
            kerning = self._face.get_kerning(prev_char, cur_char)
            pen.x += kerning.x
            slot = self._face.glyph
            bitmap = slot.bitmap

            cur_pen.x = pen.x
            cur_pen.y = pen.y - slot.bitmap_top * 64
            self.draw_ft_bitmap(image, bitmap, cur_pen, color)

            pen.x += int(gap * slot.advance.x)
            prev_char = cur_char

        return image

    def draw_ft_bitmap(self, img, bitmap, pen, color):
        '''
        draw each char
        :param bitmap: bitmap
        :param pen:    pen
        :param color:  pen color e.g.(0,0,255) - red
        :return:       image
        '''
        x_pos = pen.x >> 6
        y_pos = pen.y >> 6
        cols = bitmap.width
        rows = bitmap.rows

        glyph_pixels = bitmap.buffer

        for row in range(rows):
            for col in range(cols):
                if glyph_pixels[row * cols + col] != 0:
                    img[y_pos + row][x_pos + col][0] = color[0]
                    img[y_pos + row][x_pos + col][1] = color[1]
                    img[y_pos + row][x_pos + col][2] = color[2]


class gen_id_card(object):
    def __init__(self, height=20, width=140):

        self.height = height
        self.width = width

        # self.words = open('AllWords.txt', 'r').read().split(' ')
        self.number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.char_set = self.number
        # self.char_set = self.words + self.number
        self.len = len(self.char_set)

        self.max_size = 11

        # self.font = ['data/font/msyhbd.ttf', 'data/font/times.ttf']
        self.font = ['data/font/MSYH.TTC']

    # 随机生成字串，长度固定
    # 返回text,及对应的向量
    def random_text(self):
        text = ''
        vecs = np.zeros((self.max_size * self.len))
        # size = random.randint(1, self.max_size)
        size = self.max_size
        for i in range(size):
            c = random.choice(self.char_set)
            vec = self.char2vec(c)
            text = text + c
            vecs[i * self.len:(i + 1) * self.len] = np.copy(vec)
        return text, vecs

    # 根据生成的text，生成image,返回标签和图片元素数据
    def gen_image(self, text_size=20):

        self.ft = put_chinese_text(self.font[random.randint(0, len(self.font) - 1)])
        text, vec = self.random_text()
        img = np.zeros([self.height, self.width, 3])
        color_ = (255, 255, 255)  # Write
        pos = (10, -2)
        image = self.ft.draw_text(img, pos, text, text_size, color_)
        # 仅返回单通道值，颜色对于汉字识别没有什么意义
        if self.ft.ttf == "data/font/msyhbd.ttf":
            img = common.erode_(image[:, :, 0], ksize=(3, 3))
        elif self.ft.ttf == "data/font/times.ttf":
            img = common.erode_(image[:, :, 0], ksize=(2, 2))
        else:
            img = image[:, :, 0]
        # noise
        img = common.addGaussianNoise(img, 50, 50)  # 高斯噪声
        img = common.SaltAndPepper(img, 0.2)  # 再添加10%的椒盐噪声
        return img[:, :], text, vec

    # 单字转向量
    def char2vec(self, c):
        vec = np.zeros((self.len))
        for j in range(self.len):
            if self.char_set[j] == c:
                vec[j] = 1
        return vec

    # 向量转文本
    def vec2text(self, vecs):
        text = ''
        v_len = len(vecs)
        for i in range(v_len):
            if (vecs[i] == 1):
                text = text + self.char_set[i % self.len]
        return text


if __name__ == '__main__':
    genObj = gen_id_card(height=20, width=140)
    image_data, label, vec = genObj.gen_image(text_size=18)
    cv2.imwrite("data/cut/_902.jpg", image_data)
    cv2.imshow('image', image_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
