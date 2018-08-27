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


def predict_model(model, input_):
    pred_ = model.predict(input_)
    shape = pred_[:, :, :].shape
    ctc_decode = K.ctc_decode(pred_[:, :, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
    output_ = K.get_value(ctc_decode)[:, :ocr.MAX_CAPTCHA]
    return output_


def get_data(path="data/cut/", image_height=20, image_width=140):
    data = []
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        img = cv2.imread(file_path)
        img = common.resize_(img, width=image_width, height=image_height)
        img = common.bgr2gray_(img)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -10)
        data.append(img[:, :, np.newaxis])
    return data, None


if __name__ == '__main__':

    model = ocr.build_network(image_width=None)

    weight_file = 'ctc/ocr_ctc_weights.h5'
    if os.path.exists(weight_file):
        model.load_weights(weight_file)
        basemodel = Model(inputs=model.get_layer('the_input').output, outputs=model.get_layer('dense_1').output)
        input_, output_ = get_data()
        pred_ = predict_model(basemodel, input_)
        print("input_:", input_.shape)
        print("pred_:", pred_.shape)
