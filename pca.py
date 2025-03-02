# Re-import necessary libraries after execution state reset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate sample data: 3D data points
np.random.seed(42)
X_pca = np.random.rand(100, 3) * 10  # 100 data points in 3D space

# Apply PCA to reduce from 3D to 2D
pca = PCA(n_components=2)
X_pca_2D = pca.fit_transform(X_pca)

# Create the plot
plt.figure(figsize=(8, 5))
plt.scatter(X_pca_2D[:, 0], X_pca_2D[:, 1], color='purple', alpha=0.6, edgecolors='k')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Reducing 3D Data to 2D While Keeping Key Patterns")
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.show()

