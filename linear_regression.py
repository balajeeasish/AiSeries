import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Example data: House size (sq ft) vs Price ($1000s)
X = np.array([500, 1000, 1500, 2000, 2500]).reshape(-1, 1)
y = np.array([100, 200, 250, 300, 450])

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict values
X_pred = np.linspace(400, 2600, 100).reshape(-1, 1)
y_pred = model.predict(X_pred)

# Plot the data
plt.scatter(X, y, color='blue', label="Actual Prices")
plt.plot(X_pred, y_pred, color='red', linestyle="--", label="Best-Fit Line")
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price ($1000s)")
plt.title("Linear Regression - House Price Prediction")
plt.legend()
plt.show()
