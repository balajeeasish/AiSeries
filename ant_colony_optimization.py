import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  

# Create a simple Ant Colony Optimization visualization
plt.figure(figsize=(8, 6))
G = nx.Graph()

# Define nodes (cities/locations)
locations = ["Start", "A", "B", "C", "D", "E", "End"]

# Add nodes to the graph
for loc in locations:
    G.add_node(loc)

# Define edges (paths) with initial pheromone trails (weights)
edges = [
    ("Start", "A", 5), ("Start", "B", 2), ("A", "C", 3), ("B", "C", 4),
    ("A", "D", 7), ("C", "E", 6), ("D", "E", 5), ("E", "End", 3)
]

# Add edges to the graph with weights
for edge in edges:
    G.add_edge(edge[0], edge[1], weight=edge[2])

# Get positions for the graph layout
pos = {
    "Start": (0, 2), "A": (1, 3), "B": (1, 1), "C": (2, 2), 
    "D": (3, 3), "E": (3, 1), "End": (4, 2)
}

# Draw the graph with pheromone intensity as edge width
edge_weights = [d["weight"] for _, _, d in G.edges(data=True)]
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', edge_color='gray', font_size=10, font_weight="bold")
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): w for u, v, w in edges}, font_size=10)

plt.title("Ant Colony Optimization - Finding the Best Path")
plt.show()

