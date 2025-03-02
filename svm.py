import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Generate sample data: Classifying Cats vs. Dogs based on two features
np.random.seed(42)
X_pets = np.vstack((np.random.normal(loc=[2, 2], scale=0.5, size=(50, 2)),
                    np.random.normal(loc=[6, 6], scale=0.5, size=(50, 2))))
y_pets = np.hstack((np.zeros(50), np.ones(50)))  # 0 = Cats, 1 = Dogs

# Train SVM Model with Linear Kernel
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_pets, y_pets)

# Create a mesh grid for visualization
xx, yy = np.meshgrid(np.linspace(X_pets[:, 0].min()-1, X_pets[:, 0].max()+1, 100),
                     np.linspace(X_pets[:, 1].min()-1, X_pets[:, 1].max()+1, 100))
Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the data and decision boundary
plt.figure(figsize=(8, 5))
plt.scatter(X_pets[:, 0], X_pets[:, 1], c=y_pets, cmap='coolwarm', edgecolors='k', alpha=0.7, label="Data Points")
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black', label="Decision Boundary")
plt.xlabel("Feature 1 (e.g., Ear Size)")
plt.ylabel("Feature 2 (e.g., Tail Length)")
plt.title("Support Vector Machines (SVM) - Classifying Cats vs. Dogs")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Show the plot
plt.show()

