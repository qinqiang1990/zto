# -*- coding: utf-8 -*-
import cv2

import numpy as np
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Flatten, Conv2D, Bidirectional, GRU, Lambda, BatchNormalization, Activation, add, \
    concatenate
from keras.layers import MaxPooling2D, Permute, TimeDistributed, Dropout
import keras.backend.tensorflow_backend as K
import os
import sys

sys.path.append(os.getcwd())
from config import mod_config

MAX_CAPTCHA = 11
CHAR_SET_LEN = 10

nb_filters = 32
pool_size = (2, 2)
kernel_size = (5, 5)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def build_network(image_height=128, image_width=32):
    hidden_unit = int(mod_config.getConfig("train", "hidden_unit"))

    inputs = Input(shape=(image_height, image_width, 1), name='the_input')

    x = Conv2D(64, (3, 3), strides=1, padding="same", kernel_initializer='he_normal', name="C0")(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, name="P0")(x)

    x = Conv2D(128, (3, 3), strides=1, padding="same", kernel_initializer='he_normal', name="C1")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, name="P1")(x)

    x = Conv2D(256, (3, 3), strides=1, padding="same", kernel_initializer='he_normal', name="C2")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), strides=1, padding="same", kernel_initializer='he_normal', name="C3")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 1), name="P2")(x)
    x = Dropout(0.5)(x)

    x = Conv2D(512, (3, 3), strides=1, padding="same", kernel_initializer='he_normal', name="C4")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding="same", kernel_initializer='he_normal', name="C5")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 1), name="P3")(x)

    x = Conv2D(512, (2, 2), strides=1, padding="same", kernel_initializer='he_normal', name="C6")(x)  # padding="valid")
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Permute((2, 1, 3))(x)
    x = Dropout(0.5)(x)
    x = TimeDistributed(Flatten())(x)

    # x = Bidirectional(GRU(hidden_unit, return_sequences=True, kernel_initializer='he_normal'), merge_mode='sum')(x)
    # x = BatchNormalization()(x)
    #
    # x = Bidirectional(GRU(hidden_unit, return_sequences=True, kernel_initializer='he_normal'), merge_mode='concat')(x)
    # x = BatchNormalization()(x)

    # RNN layer
    gru_1a = GRU(hidden_unit, return_sequences=True, kernel_initializer='he_normal', name='gru_1a')(x)
    gru_1b = GRU(hidden_unit, return_sequences=True, kernel_initializer='he_normal', name='gru_1b',
                 go_backwards=True)(x)
    gru_1merged = add([gru_1a, gru_1b])
    x = BatchNormalization()(gru_1merged)
    x = Dropout(0.5)(x)

    gru_2a = GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru_2a')(x)
    gru_2b = GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru_2b',
                 go_backwards=True)(x)

    gru_2a = add([gru_2a, x])
    gru_2b = add([gru_2b, x])
    gru2_merged = concatenate([gru_2a, gru_2b])

    x = BatchNormalization()(gru2_merged)

    x = Dense(CHAR_SET_LEN + 1, kernel_initializer='he_normal')(x)
    y_pred = Activation('softmax', name='softmax')(x)

    labels = Input(name='the_labels', shape=[MAX_CAPTCHA], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

    model.summary()

    return model


def decode_model_output(output, blank=10):
    batch_size, width, num_class = output.shape
    output = np.argmax(output, axis=-1)
    result = []

    for sample in output:
        sample_result = []
        sample_result.append(sample[0])
        for i in range(1, width):
            if sample[i] != sample[i - 1]:
                sample_result.append(sample[i])
        sample_result = np.asarray(sample_result, dtype=np.int32)
        result.append(sample_result)
    # filter blank
    decoded_pred_label = [sample[sample != blank].tolist() for sample in result]
    return decoded_pred_label


def test_model(model, X_test, Y_test):
    print("X_test:", X_test.shape)
    print("Y_test:", Y_test.shape)

    y_pred = model.predict(X_test)
    # shape = y_pred[:, :, :].shape

    # ctc_decode = K.ctc_decode(y_pred[:, :, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
    # out = K.get_value(ctc_decode)[:, :MAX_CAPTCHA]

    # accur = np.sum(abs(out - Y_test), axis=1)
    # accur_score = len(accur[accur == 0]) * 1.0 / len(accur)
    # print("accur_score:", accur_score)

    print(decode_model_output(y_pred) - Y_test)


def get_data(path="data/true_image/", equalize=1, label_length=11):
    files = os.listdir(path)
    data = []
    label = []
    for file in files:
        label_ = list(map(int, file.split('.')[0].split('_')[0]))
        if len(label_) == label_length:
            file_path = os.path.join(path, file)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            if equalize == 1:
                img = cv2.equalizeHist(img)
            data.append(img[:, :, np.newaxis])
            label.append(label_)

            # 旋转180度
            h_, w_ = img.shape[:2]
            M = np.array([[-1.0, 0.0, w_ - 1], [0.0, -1.0, h_ - 1]])
            img = cv2.warpAffine(img, M, (w_, h_))
            data.append(img[:, :, np.newaxis])
            label.append(label_)

    return np.array(data), np.array(label)


if __name__ == '__main__':

    new_data = int(mod_config.getConfig("train", "new_data"))
    data_set = int(mod_config.getConfig("train", "data_set"))
    testing = int(mod_config.getConfig("train", "testing"))
    equalize = int(mod_config.getConfig("train", "equalize"))

    height = int(mod_config.getConfig("train", "img_height"))
    width = int(mod_config.getConfig("train", "img_width"))

    MAX_CAPTCHA = MAX_CAPTCHA

    X_train, Y_train = get_data(path="data/true_image", equalize=equalize, label_length=MAX_CAPTCHA)

    test_size = int(X_train.shape[0] * 0.01)

    X_test = X_train[0:test_size, :, :, :]
    Y_test = Y_train[0:test_size, :]

    X_train = X_train[test_size:, :, :, :]
    Y_train = Y_train[test_size:, :]

    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)

    input_length = np.ones([X_train.shape[0], 1]) * int(mod_config.getConfig("train", "input_length"))
    label_length = np.ones([X_train.shape[0], 1]) * MAX_CAPTCHA

    inputs = {
        'the_input': X_train,
        'the_labels': Y_train,
        'input_length': input_length,
        'label_length': label_length
    }

    outputs = {'ctc': np.zeros([X_train.shape[0]])}

    img_height = int(mod_config.getConfig("train", "img_height"))
    img_width = int(mod_config.getConfig("train", "img_width"))

    model = build_network(image_height=img_height, image_width=img_width)

    weight_file = mod_config.getConfig("train", "weight_file")

    if os.path.exists(weight_file):
        model.load_weights(weight_file)
    #    test_model(model, X_test, Y_test)

    early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=1)
    checkpoint = ModelCheckpoint(filepath='./checkpoint/LSTM+BN5--{epoch:02d}--{val_loss:.3f}.hdf5',
                                 monitor='loss', verbose=1, mode='min', period=5)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    epochs = int(mod_config.getConfig("train", "epochs"))

    model.fit(inputs, outputs,
              batch_size=256,
              epochs=epochs,
              callbacks=[checkpoint],
              verbose=2,
              validation_split=0.2)

# test_model(model, X_test, Y_test)
