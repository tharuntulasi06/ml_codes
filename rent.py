import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
locations = ['A','B','C','D','E','F','G','H','I','J']
rent = [12000,15000,10000,20000,18000,22000,13000,17000,16000,21000]

# Convert locations to numbers
x = np.arange(len(locations)).reshape(-1,1)
y = np.array(rent)

model = LinearRegression()
model.fit(x, y)

# Predict
new_location_index = 10  # for 'K'
predicted_rent = model.predict([[new_location_index]])

print(f"Predicted rent: ₹{predicted_rent[0]:.2f}")
