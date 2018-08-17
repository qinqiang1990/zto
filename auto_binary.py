import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt

from telephone import hand_write

img = hand_write.get_img(str="18852890100", run=True, font_path=None)
cv2.imwrite("data/cut/_901.jpg", img)


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


path = "data/cut/"
files = os.listdir(path)
i = 1
plt.figure()
for file in files:
    file_path = os.path.join(path, file)
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    # _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, -10)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -10)
    # img = bin_(img, method="mean", rate=0.9, bais=5)
    plt.subplot(4, 5, i)
    plt.title(file)
    plt.imshow(img)
    i = i + 1
plt.show()
