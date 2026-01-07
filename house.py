import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 2: Create the Dataset
locations = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',]
rent = [12000, 15000, 10000, 20000, 18000, 22000, 13000, 17000, 16000, 21000]

x = np.arange(len(locations)).reshape(-1,1)
y = np.array(rent)

#train-test split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



model = LinearRegression()
model.fit(x, y)


# Predict
new_location = 11  # Predicting for location 'K'
predicted_rent = model.predict([[new_location]])

print(f"Predicted rent for location {new_location}: {predicted_rent[0]:.2f}")


plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label='Actual Rent', color='blue')
plt.scatter(locations, rent, color='blue')
plt.plot(x, model.predict(x), color='green', label='Regression Line')

plt.plot(new_location, predicted_rent, 'ro', label='Predicted Rent for K')
plt.xlabel('Location Index')
plt.ylabel('Rent (₹)')
plt.title('Rent Prediction based on Location')
plt.legend()
plt.tight_layout()
plt.savefig('rent_prediction.png')
plt.show()
