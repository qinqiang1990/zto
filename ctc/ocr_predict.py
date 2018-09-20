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
    output_ = K.get_value(ctc_decode)
    # return output_[:, :ocr.MAX_CAPTCHA]
    return output_


def get_data(path="./data/true_image/", image_height=32, equalize=1):
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        origin = cv2.imread(file_path)

        img = common.bgr2gray_(origin)
        h, w = img.shape[:2]

        img = cv2.resize(img, (int(w / h * image_height), image_height), interpolation=cv2.INTER_AREA)
        if equalize == 1:
            img = cv2.equalizeHist(img)

        data = img[np.newaxis, :, :, np.newaxis]
        label = list(map(int, file.split('.')[0].split('_')[0]))

        # cv2.imshow("img", img)
        # cv2.waitKey()
        yield np.array(data), np.array(label), origin, file_path


if __name__ == '__main__':

    img_height = int(mod_config.getConfig("train", "img_height"))
    equalize = int(mod_config.getConfig("train", "equalize"))

    model = ocr.build_network(image_height=img_height, image_width=None)

    weight_file = mod_config.getConfig("train", "weight_file")
    #path = "../text-detection-ctpn/data/data_bak/"
    path="./data/true_image/"
    # path = "./data/cut/"
    if os.path.exists(weight_file):
        model.load_weights(weight_file)
        basemodel = Model(inputs=model.get_layer('the_input').output,
                          outputs=model.get_layer('softmax').output)

        count = 0
        pos = 0
        for data_, label_, img, file_path in get_data(path=path, image_height=img_height, equalize=equalize):
            pred_ = predict_model(basemodel, data_)
            print("==============================")
            print("orig:", label_)
            print("pred:", pred_[0])

            if np.sum(label_ == pred_[0]) == len(label_):
                pos = pos + 1
            count = count + 1
        print("acc:", pos / count)
        # if len(label_) < 5:
        #     img_name = "".join(map(str, pred_[0]))
        #     img_name = path + img_name + "_" + str(np.random.randint(0, 100)) + ".jpg"
        #     cv2.imwrite(img_name, img)
        #     os.remove(file_path)
        #     print(img_name)
