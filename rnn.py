import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Create a simple RNN visualization
plt.figure(figsize=(8, 6))
G = nx.DiGraph()

# Define layers
inputs = ['Word 1', 'Word 2', 'Word 3', 'Word 4']
hidden = ['Memory State 1', 'Memory State 2', 'Memory State 3', 'Memory State 4']
output = ['Final Prediction']

# Add nodes to the graph
for node in inputs + hidden + output:
    G.add_node(node)

# Connect input to hidden layer (Sequential Processing)
for i in range(len(inputs)):
    G.add_edge(inputs[i], hidden[i])
    if i > 0:
        G.add_edge(hidden[i - 1], hidden[i])  # Memory connection

# Connect last hidden state to output
G.add_edge(hidden[-1], output[0])

# Draw the network
pos = {
    'Word 1': (0, 2), 'Word 2': (1, 2), 'Word 3': (2, 2), 'Word 4': (3, 2),
    'Memory State 1': (0, 1), 'Memory State 2': (1, 1), 'Memory State 3': (2, 1), 'Memory State 4': (3, 1),
    'Final Prediction': (3, 0)
}

nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', edge_color='gray', font_size=10, font_weight="bold")
plt.title("Recurrent Neural Network (RNN) - Processing Sequential Data")
plt.show()

