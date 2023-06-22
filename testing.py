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
print("cooks distances: ", model.cooks_distance)
cookie = model.cooks_distance
print(len(cookie))
print(type(cookie))

# Now do a test for the cooks distance categorical term...
unique_categories = np.array(['A', 'B', 'C'])
new_column = np.zeros((X.shape[0], 1), dtype=object)
# assign random category to the new column
for i in range(X.shape[0]):
    np.random.shuffle(unique_categories)  # shuffle anew each time
    new_column[i] = unique_categories[0]  # select the first one. Shuffling each time should ensure some randomness

X_cat = np.hstack((X, new_column))

# Now test the categorical option of linear regression
model2 = LinearRegression()
model2.fit(X_cat, y, category=2)
print(model2.weight_vector)
print(model2.cooks_distance)
# Now let us calculate the average cooks distance per category (A, B, C):
print("Mean Cook's Distance Category A:", np.mean(model2.cooks_distance['A']))
print("Mean Cook's Distance Category B:", np.mean(model2.cooks_distance['B']))
print("Mean Cook's Distance Category C:", np.mean(model2.cooks_distance['C']))
# THey all have relatively the same cooks distance! C is the highest though...