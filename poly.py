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
    'per_sqr_ft': np.random.randint(400, 2500, 500),
    'bedrooms': np.random.randint(1, 5, 500),
    'distance_mtrs': np.round(np.random.uniform(0.5, 10, 500), 1),
    'floors': np.random.randint(0, 20, 500),
    'age': np.random.randint(1, 30, 500)
})

base_rent = 7000
rent_per_sqr_ft = 5
rent_per_sqr_ft_squared = 0.01
rent_per_bedroom = 2000
rent_reduction = 800
rent_per_floor = 300
rent_per_age_reduction = 250

df['rent'] = (base_rent 
              + rent_per_sqr_ft * df['per_sqr_ft'] 
              + rent_per_sqr_ft_squared * (df['per_sqr_ft'] ** 2) 
              + rent_per_bedroom * df['bedrooms'] 
              - rent_reduction * df['distance_mtrs'] 
              + rent_per_floor * df['floors'] 
              - rent_per_age_reduction * df['age'])
 
print(df.head())

#prepare data for training
X = df['per_sqr_ft'].values.reshape(-1, 1)
Y = df['rent'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)

#predict and evaluate linear model
Y_pred_linear = linear_model.predict(X_test)
print("\nLinear Regression")
print(f"Rent = {linear_model.intercept_:.0f} + {linear_model.coef_[0]:.0f} * per_sqr_ft")

#polynomial regression model
poly_features = PolynomialFeatures(degree=2, include_bias=False)

#data transformation
X_train_poly = poly_features.fit_transform(X_train) 
X_test_poly = poly_features.transform(X_test)

#to see
print("\nPolynomial Features Sample:")
print(f"Original feature: {X_train[0][0]}")
print(f"Transformed features: {X_train_poly[0][0]:.0f}, Area^2: {X_train_poly[0][1]:.0f}")

#predict and evaluate polynomial model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, Y_train)
Y_pred_poly = poly_model.predict(X_test_poly)
print("\nPolynomial Regression")
print(f"Rent = {poly_model.intercept_:.0f} + {poly_model.coef_[0]:.0f} * per_sqr_ft + {poly_model.coef_[1]:.4f} * per_sqr_ft^2")

#compare models
#linear vs polynomial
#plotting
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_plot_poly = poly_features.transform(X_plot)
Y_plot_linear = linear_model.predict(X_plot)
Y_plot_poly = poly_model.predict(X_plot_poly)
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, alpha=0.5, color='orange', label='Data Points')
plt.plot(X_plot, Y_plot_linear, color='blue', label='Linear Regression')
plt.plot(X_plot, Y_plot_poly, color='red', label='Polynomial Regression (Degree 2)')
plt.xlabel('Per Square Foot')
plt.ylabel('Rent (₹)')
plt.title('Rent Prediction: Linear vs Polynomial Regression')
plt.legend()
plt.tight_layout()
plt.savefig('rent_prediction_comparison.png')
plt.show()

#R2 Scores
r2_linear = r2_score(Y_test, Y_pred_linear)
r2_poly = r2_score(Y_test, Y_pred_poly)
print(f"\nR² Score Linear Regression: {r2_linear:.4f}")
print(f"R² Score Polynomial Regression: {r2_poly:.4f}")
