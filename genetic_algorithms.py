import numpy as np
import matplotlib.pyplot as plt
import random

# Simulating Genetic Algorithm Evolution
generations = np.arange(1, 21)
best_fitness = [random.uniform(50, 100) for _ in generations]  # Simulating fitness improvement

# Creating a trend of increasing fitness values
best_fitness = sorted(best_fitness, reverse=False)

# Plot the Genetic Algorithm Evolution
plt.figure(figsize=(8, 5))
plt.plot(generations, best_fitness, marker='o', linestyle='-', color='purple', label="Best Fitness Score")
plt.xlabel("Generations")
plt.ylabel("Fitness Score")
plt.title("Genetic Algorithm - Evolution of Solutions Over Time")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.show()

