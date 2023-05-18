from multiple_linear_regression import LinearRegression
import numpy as np

X = np.column_stack((np.random.randn(1000), np.random.randn(1000)))
y = 20 + (9 * X[:, 0]) + (5 * X[:, 1]) + np.random.randn(1000)
# Testing the multiple linear regression
model = LinearRegression()
model.fit(X, y)
print(model.weight_vector)
print(model.bias_term)
