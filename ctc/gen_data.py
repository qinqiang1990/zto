# coding:utf-8
import sys
import os

import numpy as np
import cv2
import string

sys.path.append(os.getcwd())

from ctc.generator import gen_id_card
from telephone import hand_write

# characters = string.digits + string.ascii_uppercase
characters = string.digits
# print(characters)

width, height, n_class = 140, 20, len(characters)


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
        number = np.random.choice(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], n_len)
        random_text = "".join(number.astype(np.unicode))
        image_data = hand_write.get_img(str=random_text, run=True, font_path=None)
        image_data = cv2.adaptiveThreshold(image_data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -10)
        x[i, :, :, 0] = image_data
        y[i] = number
    return x, y


# batch_x, batch_y = gen(256*100)

batch_x, batch_y = gen_hand_write(512*100)

cv2.imwrite("data/cut/_901.jpg", batch_x[1])
cv2.imwrite("data/cut/_902.jpg", batch_x[2])
cv2.imwrite("data/cut/_903.jpg", batch_x[3])
cv2.imwrite("data/cut/_904.jpg", batch_x[4])
cv2.imwrite("data/cut/_905.jpg", batch_x[5])

print(batch_x.shape)
print(batch_y.shape)

np.save("ctc/X_train.npy", batch_x)
np.save("ctc/Y_train.npy", batch_y)
