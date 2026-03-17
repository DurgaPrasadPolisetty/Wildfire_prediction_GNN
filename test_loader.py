from data_loader import build_graph_snapshot

data = build_graph_snapshot("2022-01-01")

print(data)
print("Node Features Shape:", data.x.shape)
print("Edges Shape:", data.edge_index.shape)
print("Labels Shape:", data.y.shape)