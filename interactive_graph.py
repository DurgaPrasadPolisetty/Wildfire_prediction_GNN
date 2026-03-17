import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_graph

edge_index = load_graph()

edges = edge_index.numpy()

num_nodes = edges.max() + 1

degree = np.zeros(num_nodes)

for src in edges[0]:

    degree[src] += 1

plt.figure(figsize=(6,6))

plt.hist(
    degree,
    bins=20,
    color="orange"
)

plt.xlabel("Node Degree")
plt.ylabel("Frequency")

plt.title("Graph Node Degree Distribution")

plt.grid()

plt.savefig("outputs/node_degree_distribution.png", dpi=300)

print("Degree distribution saved")