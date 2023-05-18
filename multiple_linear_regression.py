import numpy as np


class LinearRegression:
    """

    """
    def __init__(self):
        self.weight_vector = None
        self.bias_term = None

    def fit(self, X_train, y_train, classes=None):
        """

        :param X_train:
        :param y_train:
        :param classes: default is None, if specified, specify column index within matrix
        :return: Nahhhhh
        """
        if classes == None:
            X_train = np.column_stack((np.ones(len(X_train[:, 0])), X_train))
            weights = np.linalg.inv(X_train.T @ X_train) @ (X_train.T @ y_train)
            self.weight_vector = weights[1:]
            self.bias_term = np.mean(weights[0])
        else:
            pass