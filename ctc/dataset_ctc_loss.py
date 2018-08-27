# -*- coding: utf-8 -*-
import numpy as np
from keras import Model
from keras.layers import Input, Dense, Flatten, Conv2D, Bidirectional, GRU, Lambda
from keras.layers import MaxPooling2D, Permute, TimeDistributed, Dropout
import keras.backend.tensorflow_backend as K
import os
import sys

sys.path.append(os.getcwd())

from config import mod_config
from ctc import gen_data

MAX_CAPTCHA = 11
CHAR_SET_LEN = 10

nb_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def build_network(image_height=140, image_width=20):
    hidden_unit = int(mod_config.getConfig("train", "hidden_unit"))

    input = Input(shape=(image_height, image_width, 1), name='the_input')

    x = Conv2D(nb_filters * 1, kernel_size, activation='relu', padding="same")(input)
    x = MaxPooling2D(pool_size=pool_size, padding="same")(x)

    x = Conv2D(nb_filters * 2, kernel_size, activation='relu', padding="same")(x)
    x = MaxPooling2D(pool_size=pool_size, padding="same")(x)

    x = Conv2D(nb_filters * 3, kernel_size, activation='relu', padding="same")(x)
    x = MaxPooling2D(pool_size=pool_size, padding="same")(x)

    x = Permute((2, 1, 3))(x)
    x = Dropout(0.5)(x)
    x = TimeDistributed(Flatten())(x)

    x = Bidirectional(GRU(hidden_unit, return_sequences=True), merge_mode='concat')(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(GRU(hidden_unit, return_sequences=True), merge_mode='sum')(x)

    y_pred = Dense(CHAR_SET_LEN + 1, activation='softmax')(x)

    basemodel = Model(inputs=input, outputs=y_pred)

    labels = Input(name='the_labels', shape=[MAX_CAPTCHA], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
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
    shape = y_pred[:, :, :].shape

    ctc_decode = K.ctc_decode(y_pred[:, :, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
    out = K.get_value(ctc_decode)[:, :MAX_CAPTCHA]

    accur = np.sum(abs(out - Y_test), axis=1)
    accur_score = len(accur[accur == 0]) * 1.0 / len(accur)
    print("accur_score:", accur_score)


#     print(decode_model_output(y_pred) - Y_test)


if __name__ == '__main__':

    gen_data.run(256 * 100)

    X_train = np.load("ctc/X_train.npy")
    Y_train = np.load("ctc/Y_train.npy")

    weight_file = 'ctc/ocr_ctc_weights.h5'

    batch_size = 256
    verbose = 2
    test_size = int(X_train.shape[0] * 0.1)

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

    if os.path.exists(weight_file):
        model.load_weights(weight_file)

    basemodel = Model(inputs=model.get_layer('the_input').output, outputs=model.get_layer('dense_1').output)

    test_model(basemodel, X_test, Y_test)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    times = 1
    while times:

        epochs = int(mod_config.getConfig("train", "epochs"))
        batch_epochs = int(mod_config.getConfig("train", "save_epochs"))

        model.fit(inputs, outputs,
                  batch_size=batch_size,
                  epochs=batch_epochs,
                  verbose=verbose,
                  validation_split=0.3)

        model.save_weights(weight_file)

        if times > epochs / batch_epochs:
            break
        print("cur_epochs:", batch_epochs * times, "epochs:", epochs)
        times = times + 1
