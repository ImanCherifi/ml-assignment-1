import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("dataset_ex1.csv", sep=";", decimal=",", header=None, names=["x", "y"])

# Temperature
x = data["x"].to_numpy()
# MW per hour
y = data["y"].to_numpy()




# ================= First iteration

# Standardize x
x_mean = np.mean(x)
x_std = np.std(x)

x_standardized = (x - x_mean) / x_std


# Learning rate
l = 0.7
# We start with θ0 = 0 and θ1 = 0
θ0_old = 0
θ1_old = 0

# Calculate predictions
preds = np.zeros(y.size)

errs1 = preds - y

# Calculate gradients


# θ0
#For θ0, x0 = 1
θ0_gr = np.mean(errs1 * 1)

# θ1
θ1_gr = np.mean(errs1 * x_standardized)

# Update formula
# θj = θj_old - l * θJ_gr

print("1st iteration")
θ0 = θ0_old - l * θ0_gr
print(f"θ0 = {θ0}")

θ1 = θ1_old - l * θ1_gr
print(f"θ1 = {θ1}")

# Model
# y = θ0 + θ1 * x
x_line = np.linspace(x_standardized.min(), x_standardized.max(), 100)
y_line = θ1 * x_line + θ0

plt.figure()
plt.scatter(x_standardized, y, color="blue", label="Data points")
plt.plot(x_line, y_line, color="red", label="Regression line IT1")
plt.xlabel("Standardized temperature")
plt.ylabel("MW per hour")
plt.legend()
# plt.savefig("gradient_descent_it1.png")
plt.show()

# Cost function
# J = SSE / 2n
J1 = 0
for error in errs1:
    J1 += pow(error, 2)
J1 = J1 / (2 * y.size)
print(f"Cost function (IT1) = {J1}")

θ0_old = θ0
θ1_old = θ1

# ================= Second iteration

# Predictions 2
preds = θ1 * x_standardized + θ0

# Errors 2
errs = preds - y

# Calculate gradients

θ0_gr = np.mean(errs * 1)

# θ1
θ1_gr = np.mean(errs * x_standardized)

# Update formula
# θj = θj_old - l * θJ_gr

print("2nd iteration")
θ0 = θ0_old - l * θ0_gr
print(f"θ0 = {θ0}")

θ1 = θ1_old - l * θ1_gr
print(f"θ1 = {θ1}")

# Model
# y = θ0 + θ1 * x
x_line = np.linspace(x_standardized.min(), x_standardized.max(), 100)
y_line = θ1 * x_line + θ0

plt.figure()
plt.scatter(x_standardized, y, color="blue", label="Data points")
plt.plot(x_line, y_line, color="red", label="Regression line IT2")
plt.xlabel("Standardized temperature")
plt.ylabel("MW per hour")
plt.legend()
# plt.savefig("gradient_descent_it2.png")
plt.show()

# Cost function
# J = SSE / 2n
J2 = 0
for error in errs:
    J2 += pow(error, 2)
J2 = J2 / (2 * y.size)
print(f"Cost function (IT2) = {J2}")

θ0_old = θ0
θ1_old = θ1



# ================= Third iteration

plt.figure()

# Predictions 3
preds = θ1 * x_standardized + θ0

# Errors 3
errs = preds - y

# Calculate gradients


θ0_gr = np.mean(errs * 1)

# θ1
θ1_gr = np.mean(errs * x_standardized)

# Update formula
# θj = θj_old - l * θJ_gr

print("3rd iteration")
θ0 = θ0_old - l * θ0_gr
print(f"θ0 = {θ0}")

θ1 = θ1_old - l * θ1_gr
print(f"θ1 = {θ1}")

# Model
# y = θ0 + θ1 * x
x_line = np.linspace(x_standardized.min(), x_standardized.max(), 100)
y_line = θ1 * x_line + θ0

plt.scatter(x_standardized, y, color="blue", label="Data points")
plt.plot(x_line, y_line, color="red", label="Regression line IT3")
plt.xlabel("Standardized temperature")
plt.ylabel("MW per hour")
plt.legend()
# plt.savefig("gradient_descent_it3.png")
plt.show()

# Cost function
# J = SSE / 2n
J3 = 0
for error in errs:
    J3 += pow(error, 2)
J3 = J3 / (2 * y.size)
print(f"Cost function (IT3) = {J3}")

θ0_old = θ0
θ1_old = θ1

# ================= Fourth iteration

plt.figure()

# Predictions 2
preds = θ1 * x_standardized + θ0


# Errors 2
errs = preds - y


# Calculate gradients


θ0_gr = np.mean(errs * 1)


# θ1
θ1_gr = np.mean(errs * x_standardized)


# Update formula
# θj = θj_old - l * θJ_gr

print("4th iteration")
θ0 = θ0_old - l * θ0_gr
print(f"θ0 = {θ0}")

θ1 = θ1_old - l * θ1_gr
print(f"θ1 = {θ1}")

# Model
# y = θ0 + θ1 * x
x_line = np.linspace(x_standardized.min(), x_standardized.max(), 100)
y_line = θ1 * x_line + θ0

plt.scatter(x_standardized, y, color="blue", label="Data points")
plt.plot(x_line, y_line, color="red", label="Regression line IT4")
plt.xlabel("Standardized temperature")
plt.ylabel("MW per hour")
plt.legend()
# plt.savefig("gradient_descent_it4.png")
plt.show()

# Cost function
# J = SSE / 2n
J4 = 0
for error in errs:
    J4 += pow(error, 2)
J4 = J4 / (2 * y.size)
print(f"Cost function (IT4) = {J4}")

θ0_old = θ0
θ1_old = θ1

# ============================== # Cost function graph

iterations = [1, 2, 3, 4]
cost = [J1, J2, J3, J4]

plt.figure(figsize=(8, 5))
plt.plot(iterations, cost, marker='o', color='b', linestyle='-', linewidth=2)

plt.title('Cost Function Over Iterations', fontsize=14, fontweight='bold')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Cost', fontsize=12)

plt.xticks(iterations)

# plt.savefig("cost_function.png")
plt.show()