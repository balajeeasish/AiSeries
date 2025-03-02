import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate sample data for customer segmentation
np.random.seed(42)
X_customers = np.vstack((np.random.normal(loc=[2, 2], scale=0.5, size=(50, 2)),
                         np.random.normal(loc=[6, 6], scale=0.5, size=(50, 2)),
                         np.random.normal(loc=[10, 2], scale=0.5, size=(50, 2))))

# Apply K-Means Clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_customers)

# Create the plot
plt.figure(figsize=(8, 5))
plt.scatter(X_customers[:, 0], X_customers[:, 1], c=clusters, cmap='viridis', edgecolors='k', alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label="Cluster Centers")
plt.xlabel("Feature 1 (e.g., Spending Score)")
plt.ylabel("Feature 2 (e.g., Annual Income)")
plt.title("K-Means Clustering - Customer Segmentation")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.show()

