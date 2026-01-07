import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'ctr': [3.2, 5.8, 2.1, 7.4, 4.5, 6.2, 1.8, 5.1, 3.9, 8.5,
            4.2, 2.8, 6.8, 3.5, 7.1, 2.4, 5.5, 4.8, 6.5, 3.1,
            7.8, 2.6, 5.3, 4.1, 6.9, 3.7, 8.1, 2.2, 5.9, 4.6],

    'total_views': [12000, 28000, 8500, 42000, 19000, 33000, 7000, 24000, 16000, 51000,
                    18000, 11000, 38000, 14500, 44000, 9500, 26000, 21000, 35000, 13000,
                    47000, 10000, 25000, 17500, 40000, 15000, 49000, 8000, 31000, 20000]
}

df = pd.DataFrame(data)

# Your code starts here...
X = df[['ctr']]
y = df['total_views']
print(X.head())
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
# Predict
arjun_new_ctr = 8.0
predicted_views = model.predict([[arjun_new_ctr]])
print(f"Predicted total views for CTR {arjun_new_ctr}%: {predicted_views[0]:.2f}")
# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(df['ctr'], df['total_views'], color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='green', label='Regression Line')
plt.scatter(arjun_new_ctr, predicted_views, color='red', label='Predicted Views for CTR 8.0%')
plt.xlabel('Click-Through Rate (CTR) %')
plt.ylabel('Total Views')
plt.title('Total Views Prediction based on CTR')
plt.legend()
plt.tight_layout()
plt.savefig('views_prediction.png')
plt.show()
