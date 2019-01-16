import numpy as np


def train_test_split(X, y):
    """Missing info"""
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


def print_shapes(X_train, X_test, y_train, y_test):
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of y_test: {y_test.shape}")
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert y_train.shape[1] == y_test.shape[1]