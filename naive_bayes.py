# Re-import necessary libraries after execution state reset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Generate sample data: Words in emails (spam vs. non-spam classification)
np.random.seed(42)
X_words = np.vstack((np.random.normal(loc=[2, 2], scale=0.5, size=(50, 2)),
                     np.random.normal(loc=[6, 6], scale=0.5, size=(50, 2))))

y_labels = np.hstack((np.zeros(50), np.ones(50)))  # 0 = Non-Spam, 1 = Spam

# Apply Naïve Bayes Classification
nb_model = GaussianNB()
nb_model.fit(X_words, y_labels)
y_pred = nb_model.predict(X_words)

# Create the plot
plt.figure(figsize=(8, 5))
plt.scatter(X_words[:, 0], X_words[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k', alpha=0.7)
plt.xlabel("Feature 1 (e.g., Word Frequency of 'Free')")
plt.ylabel("Feature 2 (e.g., Word Frequency of 'Congratulations')")
plt.title("Naïve Bayes - Spam Email Classification")
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.show()

