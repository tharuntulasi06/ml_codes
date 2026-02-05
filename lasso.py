import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error

np.random.seed(5)
N_train_sample = 12

X_train = np.sort(np.random.uniform(-3, 3, size=N_train_sample)).reshape(-1, 1)
y_true_train = 0.5 * X_train.squeeze()**3 - 2 * X_train.squeeze()
y_train = y_true_train + np.random.normal(0, 0.3, size=N_train_sample)

X_test = np.linspace(-3, 3, 200).reshape(-1, 1)
y_true_test = 0.5 * X_test.squeeze()**3 - 2 * X_test.squeeze()
y_test = y_true_test

degree = 15

linear_model = Pipeline([
    ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
    ("linreg", LinearRegression()),
])

lasso_model = Pipeline([
    ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
    ("scaler", StandardScaler()),
    ("lasso", Lasso(alpha=0.05, max_iter=50000  )),
])

linear_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)

y_predict_linear_test = linear_model.predict(X_test)
y_predict_lasso_test = lasso_model.predict(X_test)

MSE_linear_train = mean_squared_error(y_train, linear_model.predict(X_train))
MSE_lasso_train = mean_squared_error(y_train, lasso_model.predict(X_train))

MSE_linear_test = mean_squared_error(y_test, linear_model.predict(X_test))
MSE_lasso_test = mean_squared_error(y_test, lasso_model.predict(X_test))

plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, label='Training Data')
plt.plot(X_test, y_true_test, label='True Curve')
plt.plot(X_test, y_predict_linear_test, label='Linear Model Prediction')
plt.plot(X_test, y_predict_lasso_test, label='Lasso Model Prediction')
plt.savefig('lasso_regression.png')

plt.title(f'Linear vs Lasso Regression (Degree={degree})')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()