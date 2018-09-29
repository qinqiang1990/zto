import cv2
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split


def get_data(path, h=32, w=160):
    print("============", path, "===========")
    data = []
    label = []
    name = []
    files = os.listdir(path)
    for file in files:
        img = cv2.imread(path + file, 0)
        h_, w_ = img.shape[:2]
        for angle in [0, 180]:

            if angle == 0:
                label.append([1, 0])
                M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

            elif angle == 180:
                label.append([0, 1])
                M = np.array([[-1.0, 0.0, w_ - 1], [0.0, -1.0, h_ - 1]])

            temp = cv2.warpAffine(img, M, (w_, h_))
            temp = cv2.equalizeHist(temp)
            temp = cv2.resize(temp, (w, h), interpolation=cv2.INTER_AREA)
            data.append(temp.reshape(1, -1)[0, :])
            # data.append(temp[:, :, np.newaxis])
            name.append(file)

    data = np.array(data)
    label = np.array(label)

    return data, label, name


def train(path="data/", h=32, w=160):
    data, label, _ = get_data(path=path, h=h, w=w)

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.1)

    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)

    print("x_test:", x_test.shape)
    print("y_test:", y_test.shape)

    clf = MLPClassifier(hidden_layer_sizes=(500,), learning_rate_init=0.01,
                        activation="relu", solver='adam', max_iter=200)
    clf.fit(x_train, y_train)
    joblib.dump(clf, "checkpoint/clf.pkl")

    res = clf.predict(x_train)
    true_num = 0
    num = len(res)
    for i in range(num):
        if np.sum(res[i] == y_train[i]) == len(y_train[i]):
            true_num = true_num + 1

    print("Total num:", num, "True num:", true_num, " True Rate:", true_num / float(num))

    clf = joblib.load('checkpoint/clf.pkl')
    res = clf.predict(x_test)

    true_num = 0
    num = len(res)
    for i in range(num):
        if np.sum(res[i] == y_test[i]) == len(y_test[i]):
            true_num = true_num + 1
    print("Total num:", num, "True num:", true_num, " True Rate:", true_num / float(num))


def predict(path="data/", h=32, w=160):
    x_test, y_test, name = get_data(path=path, h=h, w=w)

    print("x_test:", x_test.shape)
    print("y_test:", y_test.shape)

    clf = joblib.load('checkpoint/clf.pkl')
    res = clf.predict(x_test)

    true_num = 0
    num = len(res)
    for i in range(num):
        if np.argmax(res[i]) == np.argmax(y_test[i]):
            true_num = true_num + 1
       # else:
       #     print(name[i], "true:", np.argmax(y_test[i]), "pred:", np.argmax(res[i]))

    print("Total num:", num, "True num:", true_num, " True Rate:", true_num / float(num))


if __name__ == "__main__":
    # path = "test/"
    # path = "data_cut/"
    train(path="test/", h=32, w=160)
    predict(path='test/', h=32, w=160)
