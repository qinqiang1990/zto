import numpy as np
from sklearn.externals import joblib
from direction.MLP_Classifier import get_data

if __name__ == "__main__":

    # path = "data/"
    # path = "test/"
    path = "data_cut/"
    x_test, y_test, name = get_data(path=path)

    print("x_test:", x_test.shape)
    print("y_test:", y_test.shape)

    clf = joblib.load('clf.pkl')
    res = clf.predict(x_test)

    true_num = 0
    num = len(res)
    for i in range(num):
        if np.sum(res[i] == y_test[i]) == len(y_test[i]):
            true_num = true_num + 1
        else:
            print(name[i], "true:", y_test[i], "pred:", res[i])

    print("Total num:", num, "True num:", true_num, " True Rate:", true_num / float(num))
