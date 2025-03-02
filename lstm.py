import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  # Import NetworkX

plt.figure(figsize=(8, 6))
G = nx.DiGraph()

# Define layers
inputs = ['Word 1', 'Word 2', 'Word 3', 'Word 4']
memory = ['Forget Gate', 'Memory Cell', 'Output Gate']
output = ['Final Prediction']

# Add nodes to the graph
for node in inputs + memory + output:
    G.add_node(node)

# Connect input to memory layers (Sequential Processing)
for i in range(len(inputs)):
    G.add_edge(inputs[i], memory[1])  # Connecting words to Memory Cell
G.add_edge(memory[0], memory[1])  # Forget Gate controlling memory
G.add_edge(memory[1], memory[2])  # Memory Cell sending information to Output Gate

# Connect Output Gate to final prediction
G.add_edge(memory[2], output[0])

# Define positions for nodes
pos = {
    'Word 1': (0, 2), 'Word 2': (1, 2), 'Word 3': (2, 2), 'Word 4': (3, 2),
    'Forget Gate': (1, 1), 'Memory Cell': (2, 1), 'Output Gate': (3, 1),
    'Final Prediction': (3, 0)
}

# Draw the network
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', edge_color='gray', font_size=10, font_weight="bold")
plt.title("Long Short-Term Memory (LSTM) - Remembering Sequential Data")
plt.show()
