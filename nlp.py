# Create a simple NLP pipeline visualization
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  
plt.figure(figsize=(8, 6))
G = nx.DiGraph()

# Define NLP processing steps
steps = [
    "User Input: 'I need help with my order'",
    "Tokenization: ['I', 'need', 'help', 'with', 'my', 'order']",
    "Named Entity Recognition (NER): {'order': 'Product'}",
    "Sentiment Analysis: 'Neutral'",
    "Chatbot Response: 'How can I assist you with your order?'"
]

# Add nodes to the graph
for step in steps:
    G.add_node(step)

# Connect each step to create the pipeline flow
for i in range(len(steps) - 1):
    G.add_edge(steps[i], steps[i + 1])

# Define positions for better visualization
pos = {steps[i]: (0, -i) for i in range(len(steps))}

# Draw the NLP processing flow
nx.draw(G, pos, with_labels=True, node_size=2500, node_color='lightblue', edge_color='gray', font_size=10, font_weight="bold")
plt.title("Natural Language Processing (NLP) - Understanding Text")
plt.show()

