import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

# Generate sample data: Predicting house prices based on size & location score
np.random.seed(42)
X_houses = np.vstack((np.random.normal(loc=[1000, 5], scale=200, size=(50, 2)),
                      np.random.normal(loc=[2000, 8], scale=200, size=(50, 2)),
                      np.random.normal(loc=[3000, 10], scale=200, size=(50, 2))))

y_prices = np.array([150000] * 50 + [300000] * 50 + [450000] * 50)  # Simulated house prices

# Train Gradient Boosting Model
gb_model = GradientBoostingRegressor(n_estimators=3, learning_rate=0.1, random_state=42)
gb_model.fit(X_houses, y_prices)

# Predict values
y_pred = gb_model.predict(X_houses)

# Create the plot
plt.figure(figsize=(8, 5))
plt.scatter(X_houses[:, 0], y_prices, color='blue', label="Actual House Prices", alpha=0.6)
plt.scatter(X_houses[:, 0], y_pred, color='red', label="Gradient Boosting Predictions", alpha=0.6, marker="x")
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price ($)")
plt.title("Gradient Boosting - Improving House Price Predictions Step by Step")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.show()

