import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Generate synthetic data
np.random.seed(42)
df = pd.DataFrame({
    'engine_cc': np.random.randint(800, 3000, 500),
    'mileage': np.round(np.random.uniform(10, 30, 500), 1),
    'age': np.random.randint(0, 15, 500),
    'owners': np.random.randint(0, 3, 500),
    'service_rating': np.random.randint(1, 6, 500)
})

base_price = 200000
price_per_cc = 150
price_per_cc_squared = 0.02
price_reduction_mileage = 3000
price_reduction_age = 12000
price_reduction_owner = 25000
price_increase_service = 20000

df['price'] = (base_price
               + price_per_cc * df['engine_cc']
               + price_per_cc_squared * (df['engine_cc'] ** 2)
               - price_reduction_mileage * df['mileage']
               - price_reduction_age * df['age']
               - price_reduction_owner * df['owners']
               + price_increase_service * df['service_rating'])

print(df.head())

# prepare data for training
X = df['engine_cc'].values.reshape(-1, 1)
Y = df['price'].values
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)

# predict and evaluate linear model
Y_pred_linear = linear_model.predict(X_test)
print("\nLinear Regression")
print(f"Price = {linear_model.intercept_:.0f} + {linear_model.coef_[0]:.0f} * engine_cc")

# polynomial regression model
poly_features = PolynomialFeatures(degree=2, include_bias=False)

# data transformation
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# to see
print("\nPolynomial Features Sample:")
print(f"Original feature: {X_train[0][0]}")
print(f"Transformed features: {X_train_poly[0][0]:.0f}, CC^2: {X_train_poly[0][1]:.0f}")

# predict and evaluate polynomial model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, Y_train)
Y_pred_poly = poly_model.predict(X_test_poly)

print("\nPolynomial Regression")
print(f"Price = {poly_model.intercept_:.0f} + {poly_model.coef_[0]:.0f} * engine_cc + {poly_model.coef_[1]:.4f} * engine_cc^2")

# compare models
# plotting
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_plot_poly = poly_features.transform(X_plot)

Y_plot_linear = linear_model.predict(X_plot)
Y_plot_poly = poly_model.predict(X_plot_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, alpha=0.5, label='Data Points')
plt.plot(X_plot, Y_plot_linear, label='Linear Regression')
plt.plot(X_plot, Y_plot_poly, label='Polynomial Regression (Degree 2)')
plt.xlabel('Engine CC')
plt.ylabel('Car Price (₹)')
plt.title('Car Price Prediction: Linear vs Polynomial Regression')
plt.legend()
plt.tight_layout()
plt.savefig('car_price_prediction_comparison.png')
plt.show()

# R2 Scores
r2_linear = r2_score(Y_test, Y_pred_linear)
r2_poly = r2_score(Y_test, Y_pred_poly)

print(f"\nR² Score Linear Regression: {r2_linear:.4f}")
print(f"R² Score Polynomial Regression: {r2_poly:.4f}")
