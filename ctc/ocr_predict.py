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


def get_data(path="data/cut/", image_height=20, image_width=140):
    data = []
    label = []
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        img = cv2.imread(file_path)
        img = common.resize_(img, width=image_width, height=image_height)
        img = common.bgr2gray_(img)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -10)
        data.append(img[:, :, np.newaxis])
        label.append(map(int, file.split(',')[0]))
    return np.array(data), np.array(label)


if __name__ == '__main__':

    img_height = int(mod_config.getConfig("train", "img_height"))
    img_width = int(mod_config.getConfig("train", "img_width"))

    model = ocr.build_network(image_height=img_height, image_width=None)

    weight_file = 'ctc/ocr_ctc_weights.h5'
    if os.path.exists(weight_file):
        model.load_weights(weight_file)
        basemodel = Model(inputs=model.get_layer('the_input').output, outputs=model.get_layer('dense_1').output)
        data_, label_ = get_data(image_height=img_height, image_width=img_width)
        pred_ = predict_model(basemodel, data_)
        print("label_:", label_)
        print("pred_:", pred_)
