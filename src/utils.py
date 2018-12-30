import numpy as np


def train_test_split(X, y):
    """Missind info"""
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i, elem in enumerate(X):
        if ((i + 1) % 4) == 0:
            X_test.append(elem)
            y_test.append(y[i])
        else:
            X_train.append(elem)
            y_train.append(y[i])

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
