import numpy as np
import scipy.stats as stats

class LinearRegression:
    """

    Sources:
    https://rowannicholls.github.io/python/statistics/hypothesis_testing/multiple_linear_regression.html
    https://medium.com/analytics-vidhya/f-statistic-understanding-model-significance-using-python-c1371980b796

    """
    def __init__(self):
        self.weight_vector = None
        self.bias_term = None
        self.cov_matrix = None  # This is good because tells how the ind. vars correlate w ea. other
        self.f_statistic = None
        self.t_statistics = None
        self.p_values = None  # from the t_test
        self.p_value_model = None  # from the f_statistic


    def fit(self, x_train, y_train, tails=2):
        """

        NOTE: I invert the matrix, so this will take a while if the number of samples is large
        :param X_train:
        :param y_train:
        :param tails: default is a 2 tail test
        :return: Nahhhhh
        """
        x_train = np.column_stack((np.ones(len(x_train[:, 0])), x_train))
        weights = np.linalg.inv(x_train.T @ x_train) @ (x_train.T @ y_train)
        self.weight_vector = weights[1:]
        self.bias_term = np.mean(weights[0])

        # Compute f-statistic and associated p_value
        residuals = y_train - np.dot(x_train, weights)
        RSS = np.sum(residuals**2)
        TSS = np.sum((y_train - np.mean(y_train))**2)
        dof_numerator = len(x_train[0, :]) - 1  # the degrees of freedom of the numerator of f_stat eq
        dof_denominator = len(x_train[:, 0]) - dof_numerator  # degrees of freedom of denominator of f_stat eq
        numerator = RSS/dof_numerator
        denominator = TSS/dof_denominator
        self.f_statistic = numerator/denominator
        self.p_value = 1 - stats.f.cdf(self.f_statistic, dof_numerator, dof_denominator)

        # Compute t_statistics and assocaited p_values
        mse = RSS / dof_numerator  # a.k.a variance of residuals
        self.cov_matrix = mse * np.linalg.inv(x_train.T @ x_train)
        standard_errors = np.sqrt(np.diag(self.cov_matrix))
        self.t_statistics = weights / standard_errors  # includes the t_statistic of the bias term
        self.p_values = tails * (1 - stats.t.cdf(np.abs(self.t_statistics), df=dof_denominator))


    def predict(self, x_test):
        """

        :param X_test:
        :return: predictions of the target
        """
        return np.dot(x_test, self.weight_vector) + self.bias_term



