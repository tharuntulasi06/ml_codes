import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


common_interesrts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
respnse_time_minutes = [5, 7, 4, 10, 8, 12, 6, 9, 8, 11]
age_compactable = [2, 3, 1, 4, 3, 5, 2, 4, 3, 5]

match_score = [70, 75, 65, 85, 80, 90, 68, 78, 76, 88]

x=np.array([common_interesrts,respnse_time_minutes,age_compactable]).T
y=np.array(match_score)

#y=mx+c 
#y=m1x1+m2x2+m3x3+...mnxn+c


print(x)
print(f"\n shape of x: {x.shape}")

model = LinearRegression()
model.fit(x ,y)
print("\n -----What did the model learn? -----")
print(f"\nCoefficients: {model.coef_.round(2)}")
print(f"\nCommon Interests Coefficient: {model.coef_[0].round(2)} points per unit")
print(f"\nResponse Time Coefficient: {model.coef_[1].round(2)} points per unit")
print(f"\nAge Compatability Coefficient: {model.coef_[2].round(2)} points per unit")

# Predicting match score for a new profile
new_profile = [[10, 10, 10]]
predicted_score = model.predict(new_profile)

print(f"\nPredicted Match Score for profile {new_profile}: {predicted_score[0]:.1f}")

#visualization
fig , axis =plt.subplots(1,3, figsize=(18,5))

feature_names = [common_interesrts, respnse_time_minutes, age_compactable]
names = ['Common Interests', 'Response Time (minutes)', 'Age Compatability']
colors = ['blue', 'orange', 'green']

for i , (feature_names, names, color) in enumerate(zip(feature_names, names, colors)):
    axis[i].scatter(feature_names, match_score, color = color, s = 100, alpha=0.6) 
    axis[i].set_xlabel(names, fontsize=10)
    axis[i].set_ylabel('Match Score', fontsize=10)
    axis[i].set_title(f"Match Score vs {names}", fontsize=10)
    axis[i].grid(True , alpha=0.3)

plt.suptitle('Matrimony Match Score Prediction', fontsize=16)
plt.tight_layout()
plt.savefig('match_score_prediction.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n----- End of Matrimony Match Score Prediction -----\n")

