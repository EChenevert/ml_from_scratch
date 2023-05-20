import numpy as np


class LinearRegression:
    """

    Sources:
    https://rowannicholls.github.io/python/statistics/hypothesis_testing/multiple_linear_regression.html

    """
    def __init__(self):
        self.weight_vector = None
        self.bias_term = None
        self.p_values = None
        self.f_statistic = None
        self.t_statistics = None

    def fit(self, x_train, y_train):
        """

        NOTE: I invert the matrix, so this will take a while if the number of samples is large
        :param X_train:
        :param y_train:
        :param classes: default is None, if specified, specify column index within matrix
        :return: Nahhhhh
        """
        X_train = np.column_stack((np.ones(len(x_train[:, 0])), x_train))
        weights = np.linalg.inv(X_train.T @ X_train) @ (X_train.T @ y_train)
        self.weight_vector = weights[1:]
        self.bias_term = np.mean(weights[0])

        # Compute p-values and f-statistic
        residuals = y_train - np.dot(x_train, weights)
        dof = len(x_train) - weights
        mse = np.sum(residuals**2) / dof  # a.k.a variance of residuals



    def predict(self, x_test):
        """

        :param X_test:
        :return: predictions of the target
        """
        return np.dot(x_test, self.weight_vector) + self.bias_term



