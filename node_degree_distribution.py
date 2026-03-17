import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random

from data_loader import load_graph

nodes = pd.read_csv("data/processed/nodes.csv")

edge_index = load_graph()

edges = edge_index.numpy().T

sample_size = 1000
sample_nodes = random.sample(range(len(nodes)), sample_size)

G = nx.Graph()

for i in sample_nodes:

    G.add_node(i)

for e in edges:

    if e[0] in sample_nodes and e[1] in sample_nodes:

        G.add_edge(int(e[0]), int(e[1]))

plt.figure(figsize=(8,8))

nx.draw(
    G,
    node_size=10,
    width=0.2,
    node_color="orange"
)

plt.title("Graph Structure (Sampled Nodes)")

plt.savefig("outputs/graph_structure_sample.png", dpi=300)

print("Graph structure saved")