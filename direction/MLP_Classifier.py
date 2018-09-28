import cv2
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Flatten, Conv2D, Bidirectional, GRU
from keras.layers import Lambda, BatchNormalization, Activation, add, concatenate
from keras.layers import MaxPooling2D, Permute, TimeDistributed, Dropout
import keras.backend.tensorflow_backend as K
from keras.models import Sequential
from keras import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


def get_data(path, h=32, w=160):
    data = []
    label = []
    name = []
    files = os.listdir(path)
    for file in files:
        img = cv2.imread(path + file, 0)
        img = cv2.equalizeHist(img)

        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        for angle in [0, 180]:

            if angle == 0:
                label.append([1, 0])
            elif angle == 180:
                label.append([0, 1])

            # 逆时针旋转
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))

            # data.append(rotated.reshape(1, -1)[0, :])
            data.append(rotated[:, :, np.newaxis])
            name.append(file)

    data = np.array(data)
    label = np.array(label)

    return data, label, name


def build_network(image_height=128, image_width=32):
    inputs = Input(shape=(image_height, image_width, 1), name='the_input')
    x = Conv2D(32, (3, 3), strides=1, padding="same", kernel_initializer='he_normal')(inputs)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), strides=1, padding="same", kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), strides=1, padding="same", kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), strides=1, padding="same", kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), strides=1, padding="same", kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(2)(x)
    x = Activation("softmax")(x)

    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model


if __name__ == "__main__":
    h = 32
    w = 160
    path = "data/"
    data, label, _ = get_data(path=path, h=h, w=w)

    model = build_network(image_height=h, image_width=w)
    model.compile(loss="categorical_crossentropy", optimizer='adam',
                  metrics=["accuracy"])

    early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=1)
    checkpoint = ModelCheckpoint(filepath='./checkpoint/CNN--{epoch:02d}--{val_loss:.3f}.hdf5',
                                 monitor='loss', verbose=1, mode='min', period=5)

    model.fit(data, label,
              batch_size=256,
              epochs=50,
              callbacks=[checkpoint],
              verbose=2,
              validation_split=0.2)

    # x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.1)
    #
    # print("x_train:", x_train.shape)
    # print("y_train:", y_train.shape)
    #
    # print("x_test:", x_test.shape)
    # print("y_test:", y_test.shape)
    #
    # clf = MLPClassifier(hidden_layer_sizes=(500,), learning_rate_init=0.01,
    #                     activation="relu", solver='adam', max_iter=200)
    # clf.fit(x_train, y_train)
    # joblib.dump(clf, "clf.pkl")
    #
    # res = clf.predict(x_train)
    # true_num = 0
    # num = len(res)
    # for i in range(num):
    #     if np.sum(res[i] == y_train[i]) == len(y_train[i]):
    #         true_num = true_num + 1
    #     else:
    #         print("true:", y_train[i], "pred:", res[i])
    #
    # print("Total num:", num, "True num:", true_num, " True Rate:", true_num / float(num))
    #
    # clf = joblib.load('clf.pkl')
    # res = clf.predict(x_test)
    #
    # true_num = 0
    # num = len(res)
    # for i in range(num):
    #     if np.sum(res[i] == y_test[i]) == len(y_test[i]):
    #         true_num = true_num + 1
    #     else:
    #         print("true:", y_test[i], "pred:", res[i])
    # print("Total num:", num, "True num:", true_num, " True Rate:", true_num / float(num))
