import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from data_loader import load_graph

nodes = pd.read_csv("data/processed/nodes.csv")

edge_index = load_graph()

G = nx.Graph()

for i,row in nodes.iterrows():

    G.add_node(i, pos=(row["lon"], row["lat"]))

edges = edge_index.numpy().T

for e in edges[:1000]:   # draw subset for visualization

    G.add_edge(int(e[0]), int(e[1]))

pos = nx.get_node_attributes(G, "pos")

plt.figure(figsize=(8,8))

nx.draw(
    G,
    pos,
    node_size=3,
    width=0.2,
    node_color="red"
)

plt.title("Graph Representation of Geographic Grid")

plt.savefig("outputs/graph_structure.png", dpi=300)

plt.close()

print("Graph visualization saved")