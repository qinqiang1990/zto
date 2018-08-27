# coding:utf-8
import sys
import os

import numpy as np
import cv2
import string

sys.path.append(os.getcwd())

from ctc.generator import gen_id_card
from telephone import hand_write
from config import mod_config

# characters = string.digits + string.ascii_uppercase
characters = string.digits
# print(characters)


height = int(mod_config.getConfig("train", "img_height"))
width = int(mod_config.getConfig("train", "img_width"))
n_class = len(characters)


def gen(batch_size=32, n_len=11):
    genObj = gen_id_card(height=height, width=width)
    x = np.zeros((batch_size, height, width, 1), dtype=np.uint8)
    y = np.zeros((batch_size, n_len), dtype=np.uint8)
    for i in range(batch_size):
        image_data, label, vec = genObj.gen_image(text_size=18)
        image_data = cv2.adaptiveThreshold(image_data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -10)
        x[i] = image_data[:, :, np.newaxis]
        y[i] = [int(_) for _ in label]
    return x, y


def gen_hand_write(batch_size=32, n_len=11):
    hand_write.run_()
    x = np.zeros((batch_size, height, width, 1), dtype=np.uint8)
    y = np.zeros((batch_size, n_len), dtype=np.uint8)
    for i in range(batch_size):
        if i == 0:
            number = np.array([1, 8, 8, 5, 2, 8, 9, 0, 1, 0, 0])
        elif i == 1:
            number = np.array([1, 7, 2, 2, 4, 5, 3, 7, 8, 5, 0])
        elif i == 2:
            number = np.array([1, 2, 6, 4, 3, 7, 9, 0, 4, 3, 2])
        elif i == 3:
            number = np.array([5, 3, 7, 9, 1, 6, 2, 8, 4, 9, 2])
        elif i == 4:
            number = np.array([8, 6, 4, 1, 2, 5, 8, 3, 5, 6, 8])
        else:
            number = np.random.choice(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], n_len)
        random_text = "".join(number.astype(np.unicode))
        image_data = hand_write.get_img(str=random_text, run=True, font_path=None)
        image_data = cv2.adaptiveThreshold(image_data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -10)
        image_data = cv2.resize(image_data, (width, height), interpolation=cv2.INTER_AREA)
        x[i, :, :, 0] = image_data
        y[i] = number
    return x, y


def run(batch_size=256 * 100):
    # batch_x, batch_y = gen(256*100)

    batch_x, batch_y = gen_hand_write(batch_size)

    cv2.imwrite("data/cut/" + "".join(map(str, batch_y[0])) + ".jpg", batch_x[0])
    cv2.imwrite("data/cut/" + "".join(map(str, batch_y[1])) + ".jpg", batch_x[1])
    cv2.imwrite("data/cut/" + "".join(map(str, batch_y[2])) + ".jpg", batch_x[2])
    cv2.imwrite("data/cut/" + "".join(map(str, batch_y[3])) + ".jpg", batch_x[3])
    cv2.imwrite("data/cut/" + "".join(map(str, batch_y[4])) + ".jpg", batch_x[4])

    print(batch_x.shape)
    print(batch_y.shape)

    np.save("ctc/X_train.npy", batch_x)
    np.save("ctc/Y_train.npy", batch_y)


if __name__ == '__main__':
    run(256 * 100)
