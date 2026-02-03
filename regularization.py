# ==========================================================
# IMPORT STATEMENTS
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ==========================================================
# DATA CREATION (YOUR ORIGINAL PART - UNCHANGED)
# ==========================================================

np.random.seed(42)

print("=" * 50)
print(f"Regularization Regression Comparison")
print("=" * 50)

n_samples = 100

X = np.random.randn(n_samples, 4)

true_coef = np.array([5.0, 3.0, 0.0, 0.0])

y = X @ true_coef + 0.5 * np.random.randn(n_samples)

print(f"Data created: {n_samples} samples with 4 features.")
print(f"True coefficients: {true_coef}")
print(f"Note: Features 3 and 4 should be zero (noise features).")
print("=" * 50)

# ==========================================================
# TRAIN-TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================================
# MODEL TRAINING (YOUR SAME FORMAT)
# ==========================================================

# Multiple Linear Regression
mlr = LinearRegression()
mlr.fit(X_train, y_train)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso Regression
lasso = Lasso(alpha=1.0, max_iter=10000)
lasso.fit(X_train, y_train)

# Elastic Net Regression
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
elastic_net.fit(X_train, y_train)

# ==========================================================
# COEFFICIENT COMPARISON PRINT TABLE (YOUR STYLE)
# ==========================================================

print("\nCoefficients Comparison")
print("=" * 30)
print(f"{'Feature':<12}{'True':<8}{'MLR':<8}{'Ridge':<8}{'Lasso':<8}{'ElasticNet':<8}")
print("-" * 55)

for i in range(4):
    print(f"{i + 1:<4}"
          f"{true_coef[i]:>7.2f}"
          f"{mlr.coef_[i]:>7.2f}"
          f"{ridge.coef_[i]:>7.2f}"
          f"{lasso.coef_[i]:>7.2f}"
          f"{elastic_net.coef_[i]:>7.2f}")
print("-" * 55)

# ==========================================================
# MSE COMPARISON (TEST DATA)
# ==========================================================

models = {
    "MLR": mlr,
    "Ridge": ridge,
    "Lasso": lasso,
    "ElasticNet": elastic_net,
}

print("\nMean Squared Error (MSE) Comparison (Test Set)")
print("=" * 40)

for name, model in models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name:<15}: {mse:>.4f}")

print("=" * 40)

# ==========================================================
# REGULARIZATION TUNING RANGE (FOR ERROR PLOT)
# ==========================================================

alphas = np.logspace(-2, 3, 20)  # 0.01 to 1000

ridge_train_err, ridge_test_err = [], []
lasso_train_err, lasso_test_err = [], []
elastic_train_err, elastic_test_err = [], []

for alpha in alphas:

    # Ridge
    r = Ridge(alpha=alpha)
    r.fit(X_train, y_train)
    ridge_train_err.append(mean_squared_error(y_train, r.predict(X_train)))
    ridge_test_err.append(mean_squared_error(y_test, r.predict(X_test)))

    # Lasso
    l = Lasso(alpha=alpha, max_iter=10000)
    l.fit(X_train, y_train)
    lasso_train_err.append(mean_squared_error(y_train, l.predict(X_train)))
    lasso_test_err.append(mean_squared_error(y_test, l.predict(X_test)))

    # Elastic Net
    e = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
    e.fit(X_train, y_train)
    elastic_train_err.append(mean_squared_error(y_train, e.predict(X_train)))
    elastic_test_err.append(mean_squared_error(y_test, e.predict(X_test)))

# ==========================================================
# PLOTTING (YOUR EXACT STYLE)
# ==========================================================

plt.style.use('ggplot')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#FFF7EF')

line_colors = ['#FF6B6B', '#4ECDC4', '#555555', '#AAAAAA']
line_widths = [3, 3, 2, 2]

# ============================
# PLOT 1: COEFFICIENT BAR CHART (YOUR STYLE)
# ============================

x_pos = np.arange(4)
width = 0.2

ax1.bar(x_pos - 1.5 * width, true_coef, width=width,
        label='True Coef', color=line_colors[0],
        edgecolor='black', linewidth=1.2, alpha=0.9)

ax1.bar(x_pos - 0.5 * width, ridge.coef_, width=width,
        label='Ridge', color=line_colors[1],
        edgecolor='black', linewidth=1.2, alpha=0.9)

ax1.bar(x_pos + 0.5 * width, lasso.coef_, width=width,
        label='Lasso', color=line_colors[2],
        edgecolor='black', linewidth=1.2, alpha=0.9)

ax1.bar(x_pos + 1.5 * width, elastic_net.coef_, width=width,
        label='ElasticNet', color=line_colors[3],
        edgecolor='black', linewidth=1.2, alpha=0.9)

ax1.set_ylabel("Coefficient Value", fontsize=12, color='#555555')
ax1.set_title("Coefficient Comparison", fontsize=14,
              fontweight='bold', color='#333333', pad=12)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(['F1', 'F2', 'F3', 'F4'],
                    fontsize=11, color='#555555')
ax1.legend(fontsize=10, frameon=False,
           bbox_to_anchor=(1.05, 1.0), loc='upper left')

ax1.axhline(y=0, color='#333333', linestyle='--', linewidth=1.2)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ============================
# PLOT 2: ELASTIC NET PATH (YOUR STYLE)
# ============================

l1_ratios = np.linspace(0, 1, 20)
coefs = [[] for _ in range(4)]

for l1_ratio in l1_ratios:
    model = ElasticNet(alpha=1.0, l1_ratio=l1_ratio, max_iter=10000)
    model.fit(X_train, y_train)

    for i in range(4):
        coefs[i].append(model.coef_[i])

for i in range(4):
    ax2.plot(l1_ratios, coefs[i],
             linewidth=line_widths[i],
             color=line_colors[i],
             alpha=0.9,
             label=f'F{i + 1}')

ax2.set_xlabel("L1 Ratio", fontsize=12, color='#555555')
ax2.set_ylabel("Coefficient Value", fontsize=12, color='#555555')
ax2.set_title("Elastic Net Path", fontsize=14,
              fontweight='bold', color='#333333', pad=12)
ax2.axhline(y=0, color='#333333', linestyle='--', linewidth=1.2)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(fontsize=10, frameon=False, loc='upper right')

plt.tight_layout()
plt.savefig('elastic_net_comparison.png', dpi=150,
            bbox_inches='tight', facecolor='#FFF7EF')
plt.show()
