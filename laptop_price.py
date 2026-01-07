# 1. Import Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 2. Load the Dataset
data = {
    'ram_gb': [4, 8, 4, 16, 8, 8, 4, 16, 8, 16,
               4, 8, 4, 16, 8, 8, 4, 16, 8, 16,
               4, 8, 4, 16, 8, 8, 4, 16, 8, 16],

    'storage_gb': [256, 512, 128, 512, 256, 512, 256, 1024, 256, 512,
                   128, 512, 256, 1024, 256, 512, 128, 512, 256, 1024,
                   256, 512, 128, 512, 256, 512, 256, 1024, 256, 512],

    'processor_ghz': [2.1, 2.8, 1.8, 3.2, 2.4, 3.0, 2.0, 3.5, 2.6, 3.0,
                      1.6, 2.8, 2.2, 3.4, 2.5, 2.9, 1.9, 3.1, 2.3, 3.6,
                      2.0, 2.7, 1.7, 3.3, 2.4, 3.0, 2.1, 3.5, 2.6, 3.2],

    'price_inr': [28000, 45000, 22000, 72000, 38000, 52000, 26000, 95000, 42000, 68000,
                  20000, 48000, 29000, 88000, 40000, 50000, 23000, 70000, 36000, 98000,
                  25000, 46000, 21000, 75000, 39000, 53000, 27000, 92000, 43000, 73000]
}

df = pd.DataFrame(data)


# 3. Explore the Data
print("First 5 rows:\n", df.head())
print("\nDataset Summary:\n", df.describe())

# 4. Visualize Relationships (3 Scatter Plots)
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.scatter(df['ram_gb'], df['price_inr'])
plt.xlabel("RAM (GB)")
plt.ylabel("Price (INR)")
plt.title("RAM vs Price")
plt.grid(True , alpha=0.3)

plt.subplot(1, 3, 2)
plt.scatter(df['storage_gb'], df['price_inr'])
plt.xlabel("Storage (GB)")
plt.title("Storage vs Price")
plt.grid(True , alpha=0.3)

plt.subplot(1, 3, 3)
plt.scatter(df['processor_ghz'], df['price_inr'])
plt.xlabel("Processor Speed (GHz)")
plt.title("Processor vs Price")

plt.savefig('laptop_price_relationships.png')

plt.grid(True , alpha=0.3)
plt.tight_layout()
plt.show()

# 5. Train Linear Regression Model (3 Features)
X = df[['ram_gb', 'storage_gb', 'processor_ghz']]
y = df['price_inr']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# 6. Check Coefficients (Feature Importance)

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nFeature Coefficients:\n", coefficients)


# 7. Calculate R² Score (Model Accuracy)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("\nR² Score:", round(r2, 3))

# 8. Meera’s Question – Fair Price Prediction

meera_laptop = [[16, 512, 3.2]]
meera_price = model.predict(meera_laptop)

print(
    f"\nFair price for Meera's laptop "
    f"(16GB RAM, 512GB storage, 3.2GHz CPU): "
    f"₹{meera_price[0]:,.0f}"
)

# 9. Bonus Question – Overpriced Check

bonus_laptop = [[8, 512, 2.8]]
predicted_price = model.predict(bonus_laptop)

print(f"\nPredicted price for bonus laptop: ₹{predicted_price[0]:,.0f}")

if 55000 > predicted_price[0]:
    print("Conclusion: The laptop is OVERPRICED ❌")
else:
    print("Conclusion: The laptop price is FAIR ✅")
