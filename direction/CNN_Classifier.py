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


def get_non_numeric(path="non_numeric/", h=32, w=160, is_all=0):
    print("==========", path, "==========")
    data = []
    label = []
    name = []
    files = os.listdir(path)
    for file in files:
        img = cv2.imread(path + file, 0)
        h_, w_ = img.shape[:2]
        if is_all == 0 and w_ > 400:
            continue
        for angle in [0, 180]:

            if angle == 0:
                label.append([0, 0, 1])
                M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

            elif angle == 180:
                label.append([0, 0, 1])
                M = np.array([[-1.0, 0.0, w_ - 1], [0.0, -1.0, h_ - 1]])

            temp = cv2.warpAffine(img, M, (w_, h_))
            temp = cv2.equalizeHist(temp)
            temp = cv2.resize(temp, (w, h), interpolation=cv2.INTER_AREA)
            # data.append(temp.reshape(1, -1)[0, :])
            data.append(temp[:, :, np.newaxis])
            name.append(file)

    return data, label, name


def get_data(path, h=32, w=160, is_all=0):
    data = []
    label = []
    name = []
    print("==========", path, "==========")
    dirs = os.listdir(path)
    for dir in dirs:
        if os.path.isdir(path + dir) and dir in ["0", "0_", "1", "2"]:
            files = os.listdir(path + dir)
            for file in files:
                img = cv2.imread(path + dir + "/" + file, 0)
                h_, w_ = img.shape[:2]
                if is_all == 0 and w_ > 400:
                    continue
                for angle in [0, 180]:
                    if angle == 0:

                        if dir in ["0", "0_"]:
                            label.append([1, 0, 0])
                        elif dir == "1":
                            label.append([0, 1, 0])
                        elif dir == "2":
                            label.append([0, 0, 1])

                        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

                    elif angle == 180:

                        if dir in ["0", "0_"]:
                            label.append([0, 1, 0])
                        elif dir == "1":
                            label.append([1, 0, 0])
                        elif dir == "2":
                            label.append([0, 0, 1])

                        M = np.array([[-1.0, 0.0, w_ - 1], [0.0, -1.0, h_ - 1]])

                    temp = cv2.warpAffine(img, M, (w_, h_))
                    temp = cv2.equalizeHist(temp)
                    temp = cv2.resize(temp, (w, h), interpolation=cv2.INTER_AREA)
                    # data.append(temp.reshape(1, -1)[0, :])
                    data.append(temp[:, :, np.newaxis])
                    name.append(file)

    data = np.array(data)
    label = np.array(label)

    return data, label, name


def build_network(image_height=128, image_width=32):
    inputs = Input(shape=(image_height, image_width, 1), name='the_input')
    x = Conv2D(32, (3, 3), strides=1, padding="same", kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Dropout(0.5)(x)

    x = Conv2D(64, (3, 3), strides=1, padding="same", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), strides=1, padding="same", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Conv2D(128, (3, 3), strides=1, padding="same", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), strides=1, padding="same", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(3)(x)
    x = Activation("softmax")(x)

    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model


def train(path="data/", h=32, w=160):
    data, label, _ = get_data(path=path, h=h, w=w, is_all=1)
    print(data.shape)
    print(label.shape)

    model = build_network(image_height=h, image_width=w)
    # model.load_weights("checkpoint/CNN.hdf5")

    # x = model.get_layer('dropout_4').output
    # x = Dense(3)(x)
    # x = Activation("softmax")(x)
    # model = Model(inputs=model.get_layer('the_input').output, outputs=x)

    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

    early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=1)

    checkpoint = ModelCheckpoint(filepath='./checkpoint/CNN--{epoch:02d}--{val_loss:.3f}--{val_acc:.3f}.hdf5',
                                 monitor='loss', verbose=1, mode='min', period=5)

    model.fit(data, label,
              batch_size=256,
              epochs=2000,
              callbacks=[checkpoint],
              verbose=2,
              validation_split=0.3)


def predict(path="data/", h=32, w=160):
    x_test, y_test, name = get_data(path=path, h=h, w=w, is_all=1)

    print("x_test:", x_test.shape)
    print("y_test:", y_test.shape)

    model = build_network(image_height=h, image_width=w)
    model.load_weights("checkpoint/CNN.hdf5")

    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    scores = model.evaluate(x_test, y_test, verbose=0)

    res = model.predict(x_test)

    true_num = 0
    num = len(res)
    for i in range(num):
        if np.argmax(res[i]) == np.argmax(y_test[i]):
            true_num = true_num + 1
        else:
            print(name[i], "true:", np.argmax(y_test[i]), "pred:", np.argmax(res[i]))

    print("Total num:", num, "True num:", true_num, " True Rate:", true_num / float(num))
    print("scores:", scores)


if __name__ == "__main__":
    #     path = "test/"
    #     path = "data_cut/"
    train(path="test/", h=32, w=240)
    predict(path='test_cut/', h=32, w=240)
