import shutil

import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt

from telephone import hand_write

path = "..\\text-detection-ctpn\\data\\results_bak"
files = os.listdir(path)

save_path = "..\\text-detection-ctpn\\data\\datat\\"

if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)

i = 0
for file in files:
    file_path = os.path.join(path, file)
    if os.path.isdir(file_path):
        for _ in os.listdir(file_path):
            im_path = os.path.join(file_path, _)
            img = cv2.imread(im_path)
            if 240 < img.shape[1] <= 280:
                img_name = save_path + file + "_" + str(i) + ".jpg"
                cv2.imwrite(img_name, img)
                print(img_name)
                i = i + 1
