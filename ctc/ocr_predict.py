# -*- coding: utf-8 -*-
import cv2
import keras.backend.tensorflow_backend as K
import ctc.dataset_ctc_loss as ocr
from keras import Model
import numpy as np
import os
import telephone.common as common


def predict_model(model, input_):
    pred_ = model.predict(input_)
    shape = pred_[:, :, :].shape
    ctc_decode = K.ctc_decode(pred_[:, :, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
    output_ = K.get_value(ctc_decode)[:, :ocr.MAX_CAPTCHA]
    return output_


if __name__ == '__main__':
    file_name = "../data/cut/_2.jpg"
    origin = cv2.imread(file_name)
    img = common.resize_(origin, width=140, height=20)
    img = common.bgr2gray_(img)

    cv2.imshow("image", img)

    weight_file = './ocr_ctc_weights.h5'

    model = ocr.build_network()
    input_ = img[np.newaxis, :, :, np.newaxis]

    if os.path.exists(weight_file):
        model.load_weights(weight_file)
        basemodel = Model(inputs=model.get_layer('the_input').output, outputs=model.get_layer('dense_1').output)
        output_ = predict_model(basemodel, input_)
        print("input_:", input_.shape)
        print("output_:", output_.shape)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
