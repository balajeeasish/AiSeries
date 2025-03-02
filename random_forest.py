# Re-import necessary libraries after execution state reset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

# Sample Data: Credit card approval based on income, credit score, and existing loans
X = np.array([[1, 750, 0], [1, 600, 1], [1, 720, 0], [0, 800, 0], [1, 680, 1], [0, 650, 1]])
y = np.array(["Approved", "Rejected", "Approved", "Rejected", "Rejected", "Rejected"])

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=5, criterion="gini", max_depth=3, random_state=42)
rf_model.fit(X, y)

# Plot one of the trees from the Random Forest
plt.figure(figsize=(10,6))
plot_tree(rf_model.estimators_[0], feature_names=["Stable Income", "Credit Score", "Existing Loans"], 
          class_names=["Approved", "Rejected"], filled=True, rounded=True)
plt.title("Random Forest - One Decision Tree Example")
plt.show()

