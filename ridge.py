import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# DATA CREATION(overfit data)
np.random.seed(10)

n_train = 12
X_train = np.sort(np.random.uniform(-3, 3, size=n_train)).reshape(-1, 1)

y_train = 0.5 * X_train.squeeze() ** 3 - 2 * X_train.squeeze() 

y_train = y_train + np.random.normal(0, 3, size=y_train.shape)

# Test data
X_test = np.linspace(-3, 3,200).reshape(-1, 1)
Y_true_test = 0.5 * X_test.squeeze() ** 3 - 2 * X_test.squeeze()

Y_test = Y_true_test 

#To make the model overfit
degree = 15

# Linear Regression Model
# Create pipeline
linear_model = Pipeline([
    ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),
    ('lin_reg', LinearRegression())
])
#Ridge Regression Model
# Create pipeline
ridge_model = Pipeline([
    ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),
    ('ridge_reg', Ridge(alpha=10.0))
])
# Train Linear Regression Model
linear_model.fit(X_train, y_train)
# Train Ridge Regression Model
ridge_model.fit(X_train, y_train)
# Predict
Y_pred_linear = linear_model.predict(X_test)
Y_pred_ridge = ridge_model.predict(X_test)

# Evaluate
mse_linear_train = mean_squared_error(y_train, linear_model.predict(X_train))
mse_linear_test = mean_squared_error(Y_test, Y_pred_linear)
mse_ridge_train = mean_squared_error(y_train, ridge_model.predict(X_train))
mse_ridge_test = mean_squared_error(Y_test, Y_pred_ridge)

#Plotting the both curves true vs predicted, linear vs ridge
plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, color='red', label='Training Data')
plt.plot(X_test, Y_true_test, color='green', label='True Function')
plt.plot(X_test, Y_pred_linear, color='blue', linestyle='--', label=f'Linear Regression Prediction(degree={degree})')
plt.plot(X_test, Y_pred_ridge, color='orange', linestyle='--', label=f'Ridge Regression Prediction(degree={degree}), alpha=10.0')
plt.title('Comparison of Linear Regression and Ridge Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.savefig('ridge_vs_linear_regression.png')
plt.show()

# Print MSE results
print("Mean Squared Error (MSE) Comparison")
print(f"Linear Regression - Train: {mse_linear_train:.2f}, Test: {mse_linear_test:.2f}")
print(f"Ridge Regression - Train: {mse_ridge_train:.2f}, Test: {mse_ridge_test:.2f}")
