import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("dataset_ex1.csv", sep=";", decimal=",", header=None, names=["x", "y"])

# Temperature
x = data["x"].to_numpy()
# MW per hour
y = data["y"].to_numpy()

# ============= Z-score standardization
# Standardize x
x_mean = np.mean(x)
x_std = np.std(x)

x_standardized = (x - x_mean) / x_std

# print("First 5 standardized values:")
print(x_standardized)

# ============= Least squares

# Create X matrix with 1s
arr1s = np.ones(x_standardized.size)

X_prepared = np.column_stack((x_standardized, arr1s))

# Transpose X
Xt = X_prepared.T.copy()

# Calculate XtX
XtX = Xt @ X_prepared

# Inverse XtX
XtX_inv = np.linalg.inv(XtX)

# Calculate Xty
Xty = Xt @ y

# Calculate θ
# Formula: θ = XtX_inv * Xty

θ = XtX_inv @ Xty

θ0 = θ[1]
θ1 = θ[0]

print("θ0 = ", θ0)
print("θ1 = ", θ1)

x_line = np.linspace(x_standardized.min(), x_standardized.max(), 100)
y_line = θ1 * x_line + θ0

plt.scatter(x_standardized, y, color="blue", label="Data points")
plt.plot(x_line, y_line, color="red", label="Regression line")
plt.xlabel("Standardized temperature")
plt.ylabel("MW per hour")
plt.legend()
# plt.savefig("regression_least_squares.png")
plt.show()








