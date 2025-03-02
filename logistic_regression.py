# Re-import necessary libraries after execution state reset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Sample Data: Loan approval based on income, credit score, and existing loans
X = np.array([[1, 750, 0], [1, 600, 1], [1, 720, 0], [0, 800, 0], [1, 680, 1], [0, 650, 1]])
y = np.array(["Approved", "Rejected", "Approved", "Rejected", "Rejected", "Rejected"])

# Train Decision Tree Model
model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
model.fit(X, y)

# Create the Decision Tree plot
plt.figure(figsize=(8,5))
tree.plot_tree(model, feature_names=["Stable Income", "Credit Score", "Existing Loans"], 
               class_names=["Approved", "Rejected"], filled=True, rounded=True)
plt.title("Decision Tree - Loan Approval Process")
plt.show()
