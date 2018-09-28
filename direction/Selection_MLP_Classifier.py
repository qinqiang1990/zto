import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import validation_curve
from direction.MLP_Classifier import get_data

if __name__ == "__main__":
    data, label,_ = get_data(path="data/")

    print(data.shape)
    print(label.shape)


    param_range = ['identity', 'logistic', 'tanh', 'relu']

    train_score, test_score = validation_curve(
        MLPClassifier(hidden_layer_sizes=(200,), learning_rate_init=0.01,
                      activation="relu", solver='adam', max_iter=500),
        data, label,
        param_name='activation',
        param_range=param_range, cv=10, scoring='accuracy')

    train_score = np.mean(train_score, axis=1)
    test_score = np.mean(test_score, axis=1)

    print(param_range)
    print(train_score)
    print(test_score)
