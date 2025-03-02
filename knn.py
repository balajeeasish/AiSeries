import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Generate sample data: Classifying red vs. blue points
np.random.seed(42)
X_knn = np.vstack((np.random.normal(loc=[2, 2], scale=0.5, size=(50, 2)),
                   np.random.normal(loc=[6, 6], scale=0.5, size=(50, 2))))
y_knn = np.hstack((np.zeros(50), np.ones(50)))  # 0 = Red Class, 1 = Blue Class

# Train KNN Model with K=5
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_knn, y_knn)

# Create a mesh grid for visualization
xx, yy = np.meshgrid(np.linspace(X_knn[:, 0].min()-1, X_knn[:, 0].max()+1, 100),
                     np.linspace(X_knn[:, 1].min()-1, X_knn[:, 1].max()+1, 100))
Z = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the data and decision boundary
plt.figure(figsize=(8, 5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_knn[:, 0], X_knn[:, 1], c=y_knn, cmap='coolwarm', edgecolors='k', alpha=0.7, label="Data Points")
plt.xlabel("Feature 1 (e.g., Movie Genre Preference)")
plt.ylabel("Feature 2 (e.g., Viewing History)")
plt.title("K-Nearest Neighbors (KNN) - Movie Recommendation Classification")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Show the plot
plt.show()


