# coding:utf-8
import sys
import os
import cv2
import keras.backend.tensorflow_backend as K
from keras import Model
import numpy as np

sys.path.append(os.getcwd())
from telephone import common
import ctc.dataset_ctc_loss as ocr
from config import mod_config


def predict_model(model, input_):
    pred_ = model.predict(input_)
    shape = pred_[:, :, :].shape
    ctc_decode = K.ctc_decode(pred_[:, :, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
    output_ = K.get_value(ctc_decode)[:, :ocr.MAX_CAPTCHA]
    return output_


def get_data(path="./data/cut/", image_height=32):
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        img = cv2.imread(file_path)
        img = common.bgr2gray_(img)
        h, w = img.shape[:2]

        img = cv2.resize(img, (int(w / h * image_height), image_height), interpolation=cv2.INTER_AREA)

        img = cv2.equalizeHist(img)
        _, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)

        data = img[np.newaxis, :, :, np.newaxis]
        label = list(map(int, file.split('.')[0]))
        # cv2.imshow(file, img)
        # cv2.waitKey()
        yield np.array(data), np.array(label)


if __name__ == '__main__':

    img_height = int(mod_config.getConfig("train", "img_height"))

    model = ocr.build_network(image_height=img_height, image_width=None)

    weight_file = mod_config.getConfig("train", "weight_file")

    if os.path.exists(weight_file):
        model.load_weights(weight_file)
        basemodel = Model(inputs=model.get_layer('the_input').output, outputs=model.get_layer('softmax').output)
        for data_, label_ in get_data(path="./data/true_image", image_height=img_height):
            pred_ = predict_model(basemodel, data_)
            print("==============================")
            print("orig:", label_)
            print("pred:", pred_[0])

        # res = label_ - pred_
        # min_res = res[:, 3:7]
        # print(res)
        # print(min_res)
