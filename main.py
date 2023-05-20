from multiple_linear_regression import LinearRegression
import numpy as np

X = np.column_stack((np.random.randn(1000), np.random.randn(1000)))
y = 20 + (9 * X[:, 0]) + (5 * X[:, 1]) + np.random.randn(1000)
# Testing the multiple linear regression
model = LinearRegression()
model.fit(X, y)
print("weight coefs: ", model.weight_vector)
print("bias term: ", model.bias_term)
print("covariance matrix: ", model.cov_matrix)
print("t stats: ", model.t_statistics)
print("f_ stat: ", model.f_statistic)
print("p value from f stat: ", model.p_value)
print("p_values: ", model.p_values)




