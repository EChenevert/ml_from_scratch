import numpy as np
import scipy.stats as stats
import pandas as pd

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
        self.leverage = None  # potential for a sample to influence outcome
        self.cooks_distance = None  # metric for how much as sample effects the outcome
        self.categorical_importances = None


    def fit(self, x_train, y_train, tails=2, category=None):
        """

        NOTE: I invert the matrix, so this will take a while if the number of samples is large
        :param X_train:
        :param y_train:
        :param tails: default is a 2 tail test
        :param category: an index of the column that has categorical variables
        :return: Nahhhhh
        """
        if category == None:
            x_train = np.column_stack((np.ones(len(x_train[:, 0])), x_train))
            weights = np.linalg.inv(x_train.T @ x_train) @ (x_train.T @ y_train)
            self.weight_vector = weights  # now I will include the bias term in the weight vector (first one)
            # self.bias_term = np.mean(weights[0])

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

            # Calculate teh cooks distance
            self.cooks_distance = self.calculate_cooks_distance(x_train, y_train)

        else:
            # Hold the unique categorical variables
            unique_categories = np.unique(x_train[:, category])
            # just do the linear regression
            x_train = np.column_stack((np.ones(len(x_train[:, 0])), x_train))
            weights = np.linalg.inv(x_train.T @ x_train) @ (x_train.T @ y_train)
            self.weight_vector = weights  # now I will include the bias term in the weight vector (first one)

            self.cooks_distance = {}
            for cat in unique_categories:
                mask = x_train[:, category] == cat
                x_cat = x_train[mask]
                y_cat = y_train[mask]

                # gotta add teh bias term i think
                x_cat = np.column_stack((np.ones(len(x_cat[:, 0])), x_cat))

                self.cooks_distance[cat] = self.calculate_cooks_distance(x_cat, y_cat)




    def calculate_studentized_residuals(self, x_train, y_train):
        y_pred = self.predict(x_train)
        residuals = y_train - y_pred
        mse = np.sum(residuals**2) / (x_train.shape[0] - x_train.shape[1] - 1)
        leverage = self.calculate_leverage(x_train)
        studentized_residuals = residuals / np.sqrt(mse * (1 - leverage))
        return studentized_residuals

    def calculate_leverage(self, x_train):
        hat_matrix = x_train @ np.linalg.inv(x_train.T @ x_train) @ x_train.T
        leverage = np.diagonal(hat_matrix)
        return leverage

    def calculate_cooks_distance(self, x_train, y_train):
        leverage = self.calculate_leverage(x_train)
        studentized_residuals = self.calculate_studentized_residuals(x_train, y_train)
        return (studentized_residuals**2) * leverage / (1 - leverage)

    def predict(self, x_test):
        """

        :param X_test:
        :return: predictions of the target
        """
        return np.dot(x_test, self.weight_vector)






