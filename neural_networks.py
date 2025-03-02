import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Create a simple Neural Network visualization
plt.figure(figsize=(8, 6))
G = nx.DiGraph()

# Define layers
input_nodes = ['Input 1', 'Input 2', 'Input 3']
hidden_nodes = ['Hidden 1', 'Hidden 2', 'Hidden 3', 'Hidden 4']
output_nodes = ['Output']

# Add nodes to the graph
for node in input_nodes + hidden_nodes + output_nodes:
    G.add_node(node)

# Connect input to hidden layer
for i in input_nodes:
    for h in hidden_nodes:
        G.add_edge(i, h)

# Connect hidden layer to output
for h in hidden_nodes:
    for o in output_nodes:
        G.add_edge(h, o)

# Draw the network
pos = {
    'Input 1': (0, 2), 'Input 2': (0, 1), 'Input 3': (0, 0),
    'Hidden 1': (1, 2.5), 'Hidden 2': (1, 1.5), 'Hidden 3': (1, 0.5), 'Hidden 4': (1, -0.5),
    'Output': (2, 1)
}

nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', edge_color='gray', font_size=10, font_weight="bold")
plt.title("Neural Network - Simple Architecture")
plt.show()

