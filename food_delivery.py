import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Dataset
data = {
    'distance_km': [2.5, 6.0, 1.2, 8.5, 3.8, 5.2, 1.8, 7.0, 4.5, 9.2,
                    2.0, 6.5, 3.2, 7.8, 4.0, 5.8, 1.5, 8.0, 3.5, 6.8,
                    2.2, 5.5, 4.2, 9.0, 2.8, 7.2, 3.0, 6.2, 4.8, 8.2],

    'prep_time_min': [10, 20, 8, 25, 12, 18, 7, 22, 15, 28,
                      9, 19, 11, 24, 14, 17, 6, 26, 13, 21,
                      10, 16, 14, 27, 11, 23, 12, 18, 15, 25],

    'delivery_time_min': [18, 38, 12, 52, 24, 34, 14, 45, 29, 58,
                          15, 40, 21, 50, 27, 35, 11, 54, 23, 43,
                          17, 32, 26, 56, 19, 47, 20, 37, 30, 53]
}

df = pd.DataFrame(data)

# Features & Target

X = df[['distance_km', 'prep_time_min']]
y = df['delivery_time_min']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction for New Order
vikram_new_order = {'distance_km': 7.0,'prep_time_min': 15}

predicted_delivery_time = model.predict([[vikram_new_order['distance_km'],vikram_new_order['prep_time_min']]])

print(
    f"Predicted delivery time for distance "
    f"{vikram_new_order['distance_km']} km and "
    f"prep time {vikram_new_order['prep_time_min']} min: "
    f"{predicted_delivery_time[0]:.2f} minutes"
)

# Visualization with Regression Line
plt.figure(figsize=(10, 6))

# Actual data points
plt.scatter(df['distance_km'],df['delivery_time_min'],color='blue',label='Actual Data')

# Create regression line (distance varies, prep time fixed)
distance_range = np.linspace(df['distance_km'].min(),df['distance_km'].max(),100)

fixed_prep_time = vikram_new_order['prep_time_min']

predicted_line = model.predict(
    np.column_stack((
        distance_range,
        np.full_like(distance_range, fixed_prep_time)
    ))
)

# Plot regression line
plt.plot(distance_range,
         predicted_line,
         color='green',
         label=f'Regression Line (Prep Time = {fixed_prep_time} min)')

# Predicted point
plt.scatter(vikram_new_order['distance_km'],
            predicted_delivery_time,
            color='red',
            s=100,
            label='Predicted New Order')

# Labels & title
plt.xlabel('Distance (km)')
plt.ylabel('Delivery Time (min)')
plt.title('Delivery Time Prediction using Multiple Linear Regression')
plt.legend()
plt.tight_layout()
plt.savefig('delivery_time_prediction.png')
plt.show()
